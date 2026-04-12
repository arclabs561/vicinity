//! Random projection with scalar quantization for high-dimensional ANN.
//! Combines Johnson-Lindenstrauss dimension reduction with uniform scalar quantization.
//!
//! Projects high-dimensional vectors into a lower-dimensional space, then
//! stores them as 8-bit integers. Search uses the compressed representation
//! for a fast approximate distance scan, followed by an optional re-ranking
//! step against the original vectors.
//!
//! Useful for high-dimensional embeddings (768-1536d) where full-precision
//! distance computation is expensive. Projection + quantization gives 10-50x
//! compression with modest recall loss.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.3", features = ["rp_quant"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::rp_quant::{RpQuantIndex, RpQuantParams};
//!
//! let params = RpQuantParams::default();
//! let mut index = RpQuantIndex::new(768, params)?;
//!
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Construction
//!
//! 1. Generate a random projection matrix R of shape `d x d'` using Gaussian
//!    entries (LCG RNG + Box-Muller transform).
//! 2. Project each stored vector: `p = R^T * v`, producing a `d'`-dimensional
//!    representation.
//! 3. Compute per-dimension min/max across all projected vectors.
//! 4. Quantize: `q[i] = clamp((p[i] - min[i]) * scale[i], 0, 255) as u8`.
//!
//! # Search
//!
//! 1. Project and quantize the query using the same matrix and statistics.
//! 2. Scan all quantized vectors computing L2 distance in quantized space.
//! 3. Take the top `k * rerank_factor` candidates by quantized distance.
//! 4. Re-rank those candidates with exact cosine distance on original vectors.
//! 5. Return the top k.

use crate::distance::cosine_distance_normalized;
use crate::RetrieveError;

/// RpQuant construction and search parameters.
#[derive(Clone, Debug)]
pub struct RpQuantParams {
    /// Target projection dimension `d'`. Default: 64.
    pub projected_dim: usize,
    /// Candidate multiplier for re-ranking: fetch `k * rerank_factor` candidates,
    /// then re-rank to `k`. Default: 10.
    pub rerank_factor: usize,
    /// RNG seed for the random projection matrix. Default: 42.
    pub seed: u64,
}

impl Default for RpQuantParams {
    fn default() -> Self {
        Self {
            projected_dim: 64,
            rerank_factor: 10,
            seed: 42,
        }
    }
}

/// RpQuant index.
pub struct RpQuantIndex {
    dimension: usize,
    params: RpQuantParams,
    built: bool,

    /// Original (L2-normalized) vectors for re-ranking, stored flat.
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Random projection matrix, shape `d x d'`, stored in row-major order.
    /// Applying it: `p[j] = sum_i v[i] * projection[i * d' + j]`.
    projection: Vec<f32>,

    /// Quantized projected vectors, flat, `n * d'` bytes.
    quantized: Vec<u8>,

    /// Per-dimension minimum of projected coordinates (length `d'`).
    mins: Vec<f32>,

    /// Per-dimension scale factors `255 / (max - min)` (length `d'`).
    scales: Vec<f32>,
}

impl RpQuantIndex {
    /// Create a new RpQuant index.
    pub fn new(dimension: usize, params: RpQuantParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if params.projected_dim == 0 {
            return Err(RetrieveError::InvalidParameter(
                "projected_dim must be > 0".into(),
            ));
        }
        if params.rerank_factor == 0 {
            return Err(RetrieveError::InvalidParameter(
                "rerank_factor must be > 0".into(),
            ));
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            projection: Vec::new(),
            quantized: Vec::new(),
            mins: Vec::new(),
            scales: Vec::new(),
        })
    }

    /// Add a vector by value.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector from a slice.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add after build".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            self.vectors.extend(vector.iter().map(|x| x / norm));
        } else {
            self.vectors.extend_from_slice(vector);
        }
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index: generate projection matrix, project all vectors, quantize.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let d = self.dimension;
        let dp = self.params.projected_dim;
        let n = self.num_vectors;

        // Generate d x d' Gaussian projection matrix.
        self.projection = gaussian_matrix(d, dp, self.params.seed);

        // Project all vectors: for each vector v, p[j] = sum_i v[i] * R[i*dp+j].
        let mut projected = vec![0.0f32; n * dp];
        for i in 0..n {
            let v = self.get_vector(i);
            let p = &mut projected[i * dp..(i + 1) * dp];
            for (j, pj) in p.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (vi, ri) in v.iter().zip(self.projection[j..].iter().step_by(dp)) {
                    acc += vi * ri;
                }
                *pj = acc;
            }
        }

        // Compute per-dimension min and scale.
        let mut mins = vec![f32::INFINITY; dp];
        let mut maxs = vec![f32::NEG_INFINITY; dp];
        for i in 0..n {
            let p = &projected[i * dp..(i + 1) * dp];
            for j in 0..dp {
                if p[j] < mins[j] {
                    mins[j] = p[j];
                }
                if p[j] > maxs[j] {
                    maxs[j] = p[j];
                }
            }
        }
        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let range = mx - mn;
                if range > 1e-10 {
                    255.0 / range
                } else {
                    0.0
                }
            })
            .collect();

        // Quantize all projected vectors.
        let mut quantized = vec![0u8; n * dp];
        for i in 0..n {
            let p = &projected[i * dp..(i + 1) * dp];
            let q = &mut quantized[i * dp..(i + 1) * dp];
            for j in 0..dp {
                let val = (p[j] - mins[j]) * scales[j];
                q[j] = val.clamp(0.0, 255.0) as u8;
            }
        }

        self.mins = mins;
        self.scales = scales;
        self.quantized = quantized;
        self.built = true;
        Ok(())
    }

    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// Returns `(doc_id, distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.is_empty() {
            return Err(RetrieveError::EmptyQuery);
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }
        if k == 0 {
            return Ok(Vec::new());
        }

        let dp = self.params.projected_dim;
        let n = self.num_vectors;

        // Normalize query.
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_norm: Vec<f32> = if norm > 1e-10 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        };

        // Project query.
        let pq: Vec<f32> = (0..dp)
            .map(|j| {
                query_norm
                    .iter()
                    .enumerate()
                    .map(|(i, &qi)| qi * self.projection[i * dp + j])
                    .sum()
            })
            .collect();

        // Quantize query.
        let qq: Vec<u8> = pq
            .iter()
            .zip(self.mins.iter().zip(self.scales.iter()))
            .map(|(&p, (&m, &s))| ((p - m) * s).clamp(0.0, 255.0) as u8)
            .collect();

        // Scan all quantized vectors; compute L2 in quantized space.
        let candidates_k = (k * self.params.rerank_factor).min(n);
        let mut scores: Vec<(u32, u32)> = (0..n)
            .map(|i| {
                let qv = &self.quantized[i * dp..(i + 1) * dp];
                let dist = l2_sq_u8(qv, &qq);
                (dist, i as u32)
            })
            .collect();

        // Partial sort: bring the candidates_k smallest to the front.
        scores.select_nth_unstable_by_key(candidates_k - 1, |&(d, _)| d);
        scores.truncate(candidates_k);

        // Re-rank candidates with exact cosine distance on original vectors.
        let mut reranked: Vec<(u32, f32)> = scores
            .iter()
            .map(|&(_, idx)| {
                let v = self.get_vector(idx as usize);
                let dist = cosine_distance_normalized(&query_norm, v);
                (self.doc_ids[idx as usize], dist)
            })
            .collect();

        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    // ── Internal ───────────────────────────────────────────────────────

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

/// Generate a `rows x cols` Gaussian random matrix using an LCG RNG with
/// Box-Muller transform. The resulting columns are the projection directions.
fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let n = rows * cols;
    // Gaussian pairs require an even count; pad if needed.
    let padded = if n.is_multiple_of(2) { n } else { n + 1 };
    let mut out = Vec::with_capacity(padded);

    while out.len() < padded {
        // LCG step (same multiplier as crate test suite).
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u1 = (state >> 11) as f32 / (1u64 << 53) as f32;
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f32 / (1u64 << 53) as f32;

        // Box-Muller: avoid log(0).
        let u1 = u1.max(1e-10);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        out.push(r * theta.cos());
        out.push(r * theta.sin());
    }

    out.truncate(n);
    out
}

/// L2 squared distance in quantized (u8) space.
#[inline]
fn l2_sq_u8(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as i32 - y as i32;
            (diff * diff) as u32
        })
        .sum()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..n * dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn build_and_search() {
        let dim = 32;
        let n = 50;
        let data = make_vectors(n, dim, 42);

        let mut index = RpQuantIndex::new(
            dim,
            RpQuantParams {
                projected_dim: 8,
                rerank_factor: 5,
                seed: 1,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();

        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|(id, _)| *id == 0));
    }

    #[test]
    fn self_search_recall() {
        let dim = 64;
        let n = 100;
        let data = make_vectors(n, dim, 7);

        let mut index = RpQuantIndex::new(
            dim,
            RpQuantParams {
                projected_dim: 16,
                rerank_factor: 10,
                seed: 99,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();

        let mut hits = 0;
        for i in 0..n {
            let query = &data[i * dim..(i + 1) * dim];
            let results = index.search(query, 1).unwrap();
            if results.first().map(|(id, _)| *id) == Some(i as u32) {
                hits += 1;
            }
        }
        let recall = hits as f64 / n as f64;
        assert!(
            recall > 0.5,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    #[test]
    fn dimension_reduction() {
        let dim = 128;
        let dp = 16;
        let n = 10;
        let data = make_vectors(n, dim, 55);

        let mut index = RpQuantIndex::new(
            dim,
            RpQuantParams {
                projected_dim: dp,
                rerank_factor: 3,
                seed: 2,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();

        // Quantized storage must be exactly dp bytes per vector.
        assert_eq!(index.quantized.len(), n * dp);
    }

    #[test]
    fn empty_index_errors() {
        let mut index = RpQuantIndex::new(8, RpQuantParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn dimension_mismatch() {
        let dim = 16;
        let mut index = RpQuantIndex::new(dim, RpQuantParams::default()).unwrap();

        // Wrong dimension on add.
        let result = index.add_slice(0, &[0.0f32; 8]);
        assert!(result.is_err());

        // Add valid vector, build, then wrong dimension on search.
        let data = make_vectors(5, dim, 11);
        for i in 0..5 {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        let result = index.search(&[0.0f32; 8], 1);
        assert!(result.is_err());
    }
}
