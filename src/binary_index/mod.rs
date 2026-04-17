//! Flat ANN index backed by rotation-based binary quantization.
//!
//! At build time, each vector is passed through [`BinaryQuantizer`] (random
//! orthogonal rotation + sign threshold) to produce a compact bit-packed code.
//! At search time, the asymmetric distance (float query vs binary codes) is
//! used for a full scan, and the top candidates are re-ranked against the
//! original stored vectors using exact cosine distance.
//!
//! Binary quantization gives very aggressive compression (1 bit per projected
//! dimension) at the cost of recall. The rerank step partially recovers recall
//! by evaluating `rerank_factor * k` candidates with full precision.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.6", features = ["binary_index"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::binary_index::{BinaryFlatIndex, BinaryFlatParams};
//!
//! let params = BinaryFlatParams::default();
//! let mut index = BinaryFlatIndex::new(128, params)?;
//!
//! for (id, vec) in data {
//!     index.add_slice(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```

use crate::distance::cosine_distance;
use crate::RetrieveError;
use qntz::binary::BinaryQuantizer;

/// Construction and search parameters for [`BinaryFlatIndex`].
#[derive(Clone, Debug)]
pub struct BinaryFlatParams {
    /// Number of dimensions after rotation. May be less than the input
    /// dimension for simultaneous compression. Default: same as input `dim`.
    ///
    /// When set to `0` in the default, the index constructor substitutes the
    /// actual input dimension.
    pub projected_dim: usize,
    /// Candidate multiplier for re-ranking: fetch `rerank_factor * k`
    /// candidates by binary distance, then re-rank to `k` by exact cosine.
    /// Default: 10.
    pub rerank_factor: usize,
    /// RNG seed for the rotation matrix. Default: 42.
    pub seed: u64,
}

impl Default for BinaryFlatParams {
    fn default() -> Self {
        Self {
            projected_dim: 0, // replaced by actual dim in BinaryFlatIndex::new
            rerank_factor: 10,
            seed: 42,
        }
    }
}

/// Flat scan ANN index using rotation-based binary quantization.
pub struct BinaryFlatIndex {
    dimension: usize,
    params: BinaryFlatParams,
    built: bool,

    /// Original full-precision vectors, flat row-major, for re-ranking.
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Quantizer (constructed once at `build` time).
    quantizer: Option<BinaryQuantizer>,

    /// Packed binary codes, flat. Each code is `code_len` bytes.
    codes: Vec<u8>,

    /// Bytes per code (`projected_dim.div_ceil(8)`), set at build time.
    code_len: usize,
}

impl BinaryFlatIndex {
    /// Create a new index for vectors of `dimension` dimensions.
    ///
    /// If `params.projected_dim` is 0, it is set to `dimension`.
    pub fn new(dimension: usize, mut params: BinaryFlatParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if params.rerank_factor == 0 {
            return Err(RetrieveError::InvalidParameter(
                "rerank_factor must be > 0".into(),
            ));
        }
        if params.projected_dim == 0 {
            params.projected_dim = dimension;
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            quantizer: None,
            codes: Vec::new(),
            code_len: 0,
        })
    }

    /// Add a vector by slice.
    pub fn add_slice(&mut self, id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
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
        self.vectors.extend_from_slice(vector);
        self.doc_ids.push(id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Quantize all stored vectors and prepare the index for search.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let q = BinaryQuantizer::new(self.dimension, self.params.projected_dim, self.params.seed);
        let code_len = q.code_len();
        let mut codes = Vec::with_capacity(self.num_vectors * code_len);

        for i in 0..self.num_vectors {
            let v = self.get_vector(i);
            let code = q
                .quantize(v)
                .map_err(|e| RetrieveError::InvalidParameter(format!("quantize error: {e}")))?;
            codes.extend_from_slice(&code);
        }

        self.quantizer = Some(q);
        self.codes = codes;
        self.code_len = code_len;
        self.built = true;
        Ok(())
    }

    /// Search for the `k` approximate nearest neighbors of `query`.
    ///
    /// Returns `(doc_id, distance)` pairs sorted by ascending cosine distance.
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

        let q = self
            .quantizer
            .as_ref()
            .ok_or_else(|| RetrieveError::InvalidParameter("quantizer not initialized".into()))?;
        let n = self.num_vectors;

        // Scan all codes with asymmetric distance.
        let mut scores: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let code = &self.codes[i * self.code_len..(i + 1) * self.code_len];
                // asymmetric_distance only errors on dimension mismatch, which
                // we've already validated above.
                let dist = q.asymmetric_distance(query, code).unwrap_or(f32::INFINITY);
                (dist, i)
            })
            .collect();

        // Partial sort to bring the candidates_k smallest to front.
        let candidates_k = (k * self.params.rerank_factor).min(n);
        scores.select_nth_unstable_by(candidates_k - 1, |a, b| a.0.total_cmp(&b.0));
        scores.truncate(candidates_k);

        // Re-rank candidates with exact cosine distance on original vectors.
        let mut reranked: Vec<(u32, f32)> = scores
            .iter()
            .map(|&(_, idx)| {
                let v = self.get_vector(idx);
                let dist = cosine_distance(query, v);
                (self.doc_ids[idx], dist)
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

    /// Whether the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
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
    fn build_and_search_returns_results() {
        let dim = 32;
        let n = 50;
        let data = make_vectors(n, dim, 42);

        let mut index = BinaryFlatIndex::new(
            dim,
            BinaryFlatParams {
                projected_dim: 32,
                rerank_factor: 5,
                seed: 1,
            },
        )
        .unwrap();

        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();
        assert!(!results.is_empty());
        // The query vector itself should appear in top-5.
        assert!(results.iter().any(|(id, _)| *id == 0));
    }

    #[test]
    fn self_search_recall() {
        let dim = 64;
        let n = 100;
        let data = make_vectors(n, dim, 7);

        let mut index = BinaryFlatIndex::new(
            dim,
            BinaryFlatParams {
                projected_dim: 64,
                rerank_factor: 10,
                seed: 99,
            },
        )
        .unwrap();

        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        let mut hits = 0usize;
        for i in 0..n {
            let results = index.search(&data[i * dim..(i + 1) * dim], 1).unwrap();
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
    fn projected_dim_zero_defaults_to_dim() {
        let mut index = BinaryFlatIndex::new(16, BinaryFlatParams::default()).unwrap();
        let v: Vec<f32> = (0..16).map(|i| i as f32).collect();
        index.add_slice(0, &v).unwrap();
        index.build().unwrap();
        // code_len should be 16 / 8 = 2
        assert_eq!(index.code_len, 2);
    }

    #[test]
    fn empty_index_errors_on_build() {
        let mut index = BinaryFlatIndex::new(8, BinaryFlatParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn dimension_mismatch_on_add() {
        let mut index = BinaryFlatIndex::new(16, BinaryFlatParams::default()).unwrap();
        assert!(index.add_slice(0, &[0.0f32; 8]).is_err());
    }

    #[test]
    fn dimension_mismatch_on_search() {
        let dim = 16;
        let mut index = BinaryFlatIndex::new(dim, BinaryFlatParams::default()).unwrap();
        let data = make_vectors(5, dim, 11);
        for i in 0..5 {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();
        assert!(index.search(&[0.0f32; 8], 1).is_err());
    }

    #[test]
    fn len_and_is_empty() {
        let dim = 8;
        let mut index = BinaryFlatIndex::new(dim, BinaryFlatParams::default()).unwrap();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let v = vec![1.0f32; dim];
        index.add_slice(0, &v).unwrap();
        assert!(!index.is_empty());
        assert_eq!(index.len(), 1);
    }
}
