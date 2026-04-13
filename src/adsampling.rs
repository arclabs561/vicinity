//! ADSampling: adaptive early termination for graph-based ANN search.
//!
//! Implements the distance comparison operation (DCO) from Gao & Long, SIGMOD 2023.
//! Instead of computing the full distance between a query and a candidate, ADSampling
//! evaluates dimensions in batches and applies a statistical test to reject candidates
//! early when partial evidence indicates they are far from the query.
//!
//! The technique is most effective at high dimensionality (D >= 300). At D=960 (GIST),
//! expect 3-4x speedup over exact distance; at D=128 (SIFT), expect ~1.5x.
//!
//! # Preprocessing
//!
//! A random orthogonal rotation is applied to all stored vectors and queries so that
//! partial sums over the first `d` dimensions are unbiased estimators of the full distance.
//! Without rotation, correlated dimensions (e.g., PCA-ordered) bias the estimator.
//!
//! # References
//!
//! Gao, J. & Long, C. (2023). *High-Dimensional Approximate Nearest Neighbor Search:
//! with Reliable and Efficient Distance Comparison Operations.* SIGMOD 2023.

use crate::error::RetrieveError;

/// Parameters for ADSampling search.
#[derive(Debug, Clone)]
pub struct ADSamplingParams {
    /// Confidence coefficient (epsilon_0). Higher = more conservative (fewer false
    /// rejections, less speedup). Lower = more aggressive (faster, slight recall loss).
    /// Default: 2.1 (from the paper).
    pub epsilon0: f32,

    /// Number of dimensions evaluated per batch before re-checking the threshold.
    /// Should align with SIMD width. Default: 32 (8 floats x 4 = AVX width).
    pub delta_d: usize,

    /// Seed for generating the random rotation matrix. Using the same seed produces
    /// the same rotation, ensuring reproducible results.
    pub seed: u64,
}

impl Default for ADSamplingParams {
    fn default() -> Self {
        Self {
            epsilon0: 2.1,
            delta_d: 32,
            seed: 42,
        }
    }
}

/// Precomputed state for ADSampling search.
///
/// Holds the random orthogonal rotation matrix and the rotated copy of the database
/// vectors. Wraps an existing [`HNSWIndex`](crate::hnsw::HNSWIndex) to provide
/// accelerated search via partial distance evaluation.
pub struct ADSamplingState {
    /// Row-major rotation matrix, D x D.
    rotation: Vec<f32>,
    /// Rotated database vectors, stored flat (n * D).
    rotated_vectors: Vec<f32>,
    /// Vector dimensionality.
    dimension: usize,
    /// Number of vectors.
    num_vectors: usize,
    /// Parameters.
    params: ADSamplingParams,
    /// Precomputed ratio table: ratio[i] for i in (0..D/delta_d).
    /// ratio(D, d) = (d/D) * (1 + epsilon0/sqrt(d))^2
    ratio_table: Vec<f32>,
}

impl ADSamplingState {
    /// Build ADSampling state from raw vectors.
    ///
    /// Generates a random orthogonal rotation matrix and rotates all vectors.
    /// This is a one-time O(n * D^2) preprocessing step.
    pub fn new(vectors: &[f32], dimension: usize, params: ADSamplingParams) -> Self {
        let num_vectors = vectors.len() / dimension;
        let rotation = generate_orthogonal_rotation(dimension, params.seed);
        let rotated_vectors = rotate_all(vectors, &rotation, dimension, num_vectors);

        // Precompute ratio table for each batch checkpoint.
        let num_batches = dimension / params.delta_d.max(1);
        let dim_f = dimension as f32;
        let eps = params.epsilon0;
        let mut ratio_table = Vec::with_capacity(num_batches);
        for batch_idx in 1..=num_batches {
            let d = (batch_idx * params.delta_d) as f32;
            let correction = 1.0 + eps / d.sqrt();
            ratio_table.push((d / dim_f) * correction * correction);
        }

        Self {
            rotation,
            rotated_vectors,
            dimension,
            num_vectors,
            params,
            ratio_table,
        }
    }

    /// Rotate a query vector using the stored rotation matrix.
    #[must_use]
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        rotate_vector(query, &self.rotation, self.dimension)
    }

    /// Get the rotated vector for a given internal ID.
    #[inline]
    fn rotated_vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dimension;
        &self.rotated_vectors[start..start + self.dimension]
    }

    /// Adaptive distance comparison.
    ///
    /// Returns `Some(exact_distance)` if the candidate passes (is potentially near),
    /// or `None` if the candidate is rejected (provably far given the threshold).
    ///
    /// `threshold` is the current worst distance in the result set (the k-th nearest).
    /// When the result set is not yet full, pass `f32::INFINITY` to disable early exit.
    #[inline]
    pub fn dist_comp(
        &self,
        rotated_query: &[f32],
        candidate_id: u32,
        threshold: f32,
    ) -> Option<f32> {
        let candidate = self.rotated_vector(candidate_id);
        let delta_d = self.params.delta_d;
        let dim = self.dimension;

        let mut partial_sum: f32 = 0.0;
        let mut offset = 0;

        for (batch_idx, &ratio) in self.ratio_table.iter().enumerate() {
            let end = ((batch_idx + 1) * delta_d).min(dim);
            // Accumulate squared differences for this batch.
            for i in offset..end {
                let diff = rotated_query[i] - candidate[i];
                partial_sum += diff * diff;
            }
            offset = end;

            // Statistical test: if partial distance (scaled up) exceeds threshold, reject.
            if partial_sum >= threshold * ratio {
                return None;
            }
        }

        // Computed all dimensions -- return exact distance.
        Some(partial_sum)
    }

    /// Search an HNSW index using ADSampling for accelerated distance computation.
    ///
    /// The index must already be built. Upper-layer navigation uses exact distance;
    /// base-layer beam search uses ADSampling with early termination.
    #[cfg(feature = "hnsw")]
    pub fn search_hnsw(
        &self,
        index: &crate::hnsw::HNSWIndex,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !index.is_built() {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if index.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let rotated_query = self.rotate_query(query);

        // Use a cell to track the tightest known distance (for the HNSW++ variant).
        // This gives a tighter threshold than the beam's worst distance.
        let best_k_dist = std::cell::Cell::new(f32::INFINITY);

        let dist_fn = |_q: &[f32], node_id: u32| -> f32 {
            let threshold = best_k_dist.get();
            match self.dist_comp(&rotated_query, node_id, threshold) {
                Some(exact_dist) => {
                    // Update tightest threshold.
                    if exact_dist < threshold {
                        best_k_dist.set(exact_dist);
                    }
                    exact_dist
                }
                None => {
                    // Rejected -- return a large distance so the candidate is not inserted.
                    f32::INFINITY
                }
            }
        };

        index.search_with_distance(query, k, ef, &dist_fn)
    }

    /// Number of vectors in this ADSampling state.
    #[must_use]
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Dimensionality of the vectors.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

// ─── Random orthogonal rotation ─────────────────────────────────────────────

/// Generate a D x D random orthogonal matrix via QR decomposition of a random
/// Gaussian matrix. Uses a simple LCG + Box-Muller for reproducibility.
fn generate_orthogonal_rotation(dim: usize, seed: u64) -> Vec<f32> {
    // Generate random Gaussian matrix.
    let mut rng = LcgRng::new(seed);
    let mut matrix = vec![0.0f32; dim * dim];
    for val in matrix.iter_mut() {
        *val = rng.next_gaussian() as f32;
    }

    // QR decomposition via modified Gram-Schmidt.
    let mut q = vec![0.0f32; dim * dim];
    for col in 0..dim {
        // Copy column.
        for row in 0..dim {
            q[row * dim + col] = matrix[row * dim + col];
        }

        // Orthogonalize against previous columns.
        for prev in 0..col {
            let mut dot = 0.0f32;
            for row in 0..dim {
                dot += q[row * dim + prev] * q[row * dim + col];
            }
            for row in 0..dim {
                q[row * dim + col] -= dot * q[row * dim + prev];
            }
        }

        // Normalize.
        let mut norm = 0.0f32;
        for row in 0..dim {
            norm += q[row * dim + col] * q[row * dim + col];
        }
        let norm = norm.sqrt();
        if norm > 1e-10 {
            for row in 0..dim {
                q[row * dim + col] /= norm;
            }
        }
    }

    q
}

/// Rotate a single vector: result = Q^T * v (row-major Q, so result[i] = dot(Q[*][i], v)).
fn rotate_vector(v: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotation[j * dim + i] * v[j];
        }
        result[i] = sum;
    }
    result
}

/// Rotate all vectors in a flat array.
fn rotate_all(vectors: &[f32], rotation: &[f32], dim: usize, n: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n * dim];
    for idx in 0..n {
        let src = &vectors[idx * dim..(idx + 1) * dim];
        let dst = &mut result[idx * dim..(idx + 1) * dim];
        for i in 0..dim {
            let mut sum = 0.0f32;
            for j in 0..dim {
                sum += rotation[j * dim + i] * src[j];
            }
            dst[i] = sum;
        }
    }
    result
}

/// Minimal LCG RNG for reproducible rotation matrix generation.
struct LcgRng {
    state: u64,
    has_spare: bool,
    spare: f64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            has_spare: false,
            spare: 0.0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform for Gaussian samples.
    fn next_gaussian(&mut self) -> f64 {
        if self.has_spare {
            self.has_spare = false;
            return self.spare;
        }

        loop {
            let u = self.next_uniform() * 2.0 - 1.0;
            let v = self.next_uniform() * 2.0 - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare = v * factor;
                self.has_spare = true;
                return u * factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_is_orthogonal() {
        let dim = 32;
        let q = generate_orthogonal_rotation(dim, 42);

        // Q^T * Q should be approximately I.
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += q[k * dim + i] * q[k * dim + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "Q^T*Q[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rotation_preserves_distance() {
        let dim = 64;
        let params = ADSamplingParams {
            seed: 123,
            ..Default::default()
        };
        let rotation = generate_orthogonal_rotation(dim, params.seed);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 + 1.0).collect();

        let ra = rotate_vector(&a, &rotation, dim);
        let rb = rotate_vector(&b, &rotation, dim);

        let orig_dist: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let rot_dist: f32 = ra.iter().zip(&rb).map(|(x, y)| (x - y) * (x - y)).sum();

        assert!(
            (orig_dist - rot_dist).abs() < 1e-2,
            "orig={orig_dist}, rotated={rot_dist}"
        );
    }

    #[test]
    fn dist_comp_accepts_near_neighbor() {
        let dim = 64;
        let params = ADSamplingParams::default();

        // Two nearby vectors.
        let a: Vec<f32> = vec![1.0; dim];
        let b: Vec<f32> = vec![1.01; dim];
        let vectors: Vec<f32> = [a.clone(), b].concat();

        let state = ADSamplingState::new(&vectors, dim, params);
        let rq = state.rotate_query(&a);

        // With a generous threshold, the near neighbor should be accepted.
        let result = state.dist_comp(&rq, 1, f32::INFINITY);
        assert!(result.is_some(), "near neighbor should be accepted");
    }

    #[test]
    fn dist_comp_rejects_far_candidate() {
        let dim = 128;
        let params = ADSamplingParams::default();

        let near: Vec<f32> = vec![0.0; dim];
        let far: Vec<f32> = vec![100.0; dim];
        let vectors: Vec<f32> = [near.clone(), far].concat();

        let state = ADSamplingState::new(&vectors, dim, params);
        let rq = state.rotate_query(&near);

        // With a tight threshold (distance to self = 0), the far vector should be rejected.
        let result = state.dist_comp(&rq, 1, 0.01);
        assert!(result.is_none(), "far candidate should be rejected");
    }

    #[test]
    fn exact_distance_when_no_early_exit() {
        let dim = 32; // Small dim, delta_d=32, so only one batch -- no early exit possible.
        let params = ADSamplingParams {
            delta_d: 32,
            ..Default::default()
        };

        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) + 0.5).collect();
        let vectors: Vec<f32> = [a.clone(), b.clone()].concat();

        let state = ADSamplingState::new(&vectors, dim, params);
        let rq = state.rotate_query(&a);

        // With infinite threshold, must return exact distance.
        let dist = state.dist_comp(&rq, 1, f32::INFINITY).unwrap();

        // The exact distance in rotated space should equal the original distance.
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        assert!(
            (dist - expected).abs() < 1e-1,
            "dist={dist}, expected={expected}"
        );
    }

    #[cfg(feature = "hnsw")]
    #[test]
    fn search_hnsw_with_adsampling() {
        use crate::hnsw::{HNSWIndex, HNSWParams};

        let dim = 64;
        let n = 200;
        let mut rng = LcgRng::new(99);
        let vectors: Vec<f32> = (0..n * dim).map(|_| rng.next_uniform() as f32).collect();

        // Normalize for cosine distance.
        let mut normalized = vectors.clone();
        for i in 0..n {
            let slice = &mut normalized[i * dim..(i + 1) * dim];
            let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in slice.iter_mut() {
                    *x /= norm;
                }
            }
        }

        let params = HNSWParams {
            m: 16,
            ef_construction: 100,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = HNSWIndex::with_params(dim, params).unwrap();
        let ids: Vec<u32> = (0..n as u32).collect();
        index.add_batch(&ids, &normalized).unwrap();
        let _ = index.build();

        let state = ADSamplingState::new(&normalized, dim, ADSamplingParams::default());

        let query = &normalized[0..dim];
        let results = state.search_hnsw(&index, query, 10, 50).unwrap();

        assert!(!results.is_empty());
        // The query's own vector should be in the top results (distance ~ 0).
        assert!(
            results[0].1 < 0.01,
            "self-match distance should be ~0, got {}",
            results[0].1
        );
    }
}
