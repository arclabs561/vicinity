//! SQ8U: HNSW with 8-bit scalar quantized graph traversal.
//!
//! During beam search, approximate distances are computed by decoding 8-bit
//! codes on-the-fly and accumulating squared L2 distance against the full-
//! precision query. Final top-k results are reranked with exact f32 distance.
//!
//! SQ8 is the industry-standard quantized traversal for L2 distance, used by
//! Faiss (IVFScalarQuantizer), Qdrant, and Weaviate. It offers 4x memory
//! compression with 95%+ recall on most datasets.
//!
//! Compared to SQ4U (4-bit, 8x compression, ~16 quantization levels),
//! SQ8U uses 256 quantization levels per dimension, greatly reducing
//! quantization error at the cost of 2x the code size.
//!
//! # Two-stage search
//!
//! 1. Graph traversal with SQ8 approximate L2 distance (on-the-fly decode)
//! 2. Reranking of top candidates with exact f32 distance
//!
//! # Example
//!
//! ```rust,no_run
//! # fn main() -> Result<(), vicinity::RetrieveError> {
//! use vicinity::hnsw::sq8u::HNSWSq8Index;
//!
//! let dim = 128;
//! let mut index = HNSWSq8Index::new(dim, 16, 16)?;
//!
//! let v = vec![0.1f32; dim]; // unnormalized is fine for L2
//! index.add_slice(0, &v)?;
//! // ... add more vectors ...
//!
//! index.build()?;
//!
//! // Search with quantized traversal + exact reranking
//! let q = vec![0.15f32; dim];
//! let results = index.search_reranked(&q, 10, 50, 100)?;
//! # Ok(())
//! # }
//! ```

use crate::hnsw::graph::HNSWIndex;
use crate::RetrieveError;

/// HNSW index with 8-bit scalar quantized graph traversal.
///
/// Graph construction uses full-precision f32 vectors. Search walks the graph
/// using on-the-fly SQ8 decode + L2 distance, then reranks the top candidates
/// with exact distance.
///
/// Memory: f32 vectors (for reranking) + 1 byte/dim quantized codes (4x compression).
pub struct HNSWSq8Index {
    /// The underlying HNSW index (owns graph + f32 vectors).
    index: HNSWIndex,
    /// Flat-packed 8-bit codes: codes[i * dim .. (i+1) * dim] for vector i.
    codes: Vec<u8>,
    /// Per-dimension minimum (length d).
    mins: Vec<f32>,
    /// Per-dimension step: (max - min) / 255 (length d).
    steps: Vec<f32>,
    /// Per-dimension inverse scale: 255 / (max - min) (length d).
    inv_scales: Vec<f32>,
    /// Whether quantization has been performed.
    built: bool,
}

impl HNSWSq8Index {
    /// Create a new SQ8U index with default cosine metric.
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        let index = HNSWIndex::new(dimension, m, m_max)?;
        Ok(Self {
            index,
            codes: Vec::new(),
            mins: Vec::new(),
            steps: Vec::new(),
            inv_scales: Vec::new(),
            built: false,
        })
    }

    /// Create a new SQ8U index with explicit HNSW parameters.
    pub fn with_params(
        dimension: usize,
        params: crate::hnsw::HNSWParams,
    ) -> Result<Self, RetrieveError> {
        let index = HNSWIndex::with_params(dimension, params)?;
        Ok(Self {
            index,
            codes: Vec::new(),
            mins: Vec::new(),
            steps: Vec::new(),
            inv_scales: Vec::new(),
            built: false,
        })
    }

    /// Add a vector with a document ID.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.index.add_slice(doc_id, vector)
    }

    /// Build the HNSW graph and quantize all vectors to 8-bit codes.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        self.index.build()?;
        self.quantize_vectors()?;
        self.built = true;
        Ok(())
    }

    /// Search with quantized graph traversal (no reranking).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.check_ready(query)?;
        let candidates = self.search_quantized(query, ef)?;
        let mut output: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.index.doc_ids[id as usize], dist))
            .collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output)
    }

    /// Search with oversampling + exact f32 reranking.
    ///
    /// 1. Retrieve `rerank_pool` candidates using quantized graph traversal
    /// 2. Compute exact f32 distance for each
    /// 3. Return top `k`
    pub fn search_reranked(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank_pool: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.check_ready(query)?;
        let pool = rerank_pool.max(k);
        let candidates = self.search_quantized(query, ef.max(pool))?;

        let dist_fn = self.index.dist_fn();

        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(pool)
            .map(|(internal_id, _approx)| {
                let vec = self.index.get_vector(internal_id as usize);
                let exact = dist_fn(query, vec);
                (self.index.doc_ids[internal_id as usize], exact)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.index.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.num_vectors == 0
    }

    /// Access the underlying HNSW index.
    pub fn inner(&self) -> &HNSWIndex {
        &self.index
    }

    /// Memory used by quantized codes in bytes.
    pub fn code_memory(&self) -> usize {
        self.codes.len()
    }

    // ── internal ──────────────────────────────────────────────────────────

    fn check_ready(&self, query: &[f32]) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.index.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.index.dimension,
            });
        }
        if self.index.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }
        Ok(())
    }

    /// Compute per-dimension min/max/step from the post-reorder vectors,
    /// then encode each vector as 8-bit codes.
    fn quantize_vectors(&mut self) -> Result<(), RetrieveError> {
        let dim = self.index.dimension;
        let n = self.index.num_vectors;
        let vectors = self.index.raw_vectors();

        // Compute per-dimension min/max.
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            for (d, &val) in v.iter().enumerate() {
                if val < mins[d] {
                    mins[d] = val;
                }
                if val > maxs[d] {
                    maxs[d] = val;
                }
            }
        }

        // Compute step and inv_scale.
        let mut steps = vec![0.0f32; dim];
        let mut inv_scales = vec![0.0f32; dim];
        for d in 0..dim {
            let range = maxs[d] - mins[d];
            if range > 1e-10 {
                steps[d] = range / 255.0;
                inv_scales[d] = 255.0 / range;
            }
        }

        // Encode each vector: one byte per dimension, flat-packed.
        let mut codes = vec![0u8; n * dim];
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let c = &mut codes[i * dim..(i + 1) * dim];
            for d in 0..dim {
                let q = ((v[d] - mins[d]) * inv_scales[d] + 0.5) as i32;
                c[d] = q.clamp(0, 255) as u8;
            }
        }

        self.codes = codes;
        self.mins = mins;
        self.steps = steps;
        self.inv_scales = inv_scales;
        Ok(())
    }

    /// Approximate L2^2 distance: on-the-fly decode of 8-bit codes.
    ///
    /// For each dimension: `decoded = min[d] + code * step[d]`, then
    /// accumulate `(query[d] - decoded)^2`.
    #[inline]
    fn approx_dist(query: &[f32], code: &[u8], mins: &[f32], steps: &[f32]) -> f32 {
        debug_assert_eq!(query.len(), code.len());
        let mut sum = 0.0f32;
        // Process 4 at a time for instruction-level parallelism.
        let chunks = query.len() / 4;
        let remainder = query.len() % 4;

        for i in 0..chunks {
            let base = i * 4;
            let d0 = query[base] - (mins[base] + code[base] as f32 * steps[base]);
            let d1 = query[base + 1] - (mins[base + 1] + code[base + 1] as f32 * steps[base + 1]);
            let d2 = query[base + 2] - (mins[base + 2] + code[base + 2] as f32 * steps[base + 2]);
            let d3 = query[base + 3] - (mins[base + 3] + code[base + 3] as f32 * steps[base + 3]);
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        let base = chunks * 4;
        for i in 0..remainder {
            let d = query[base + i] - (mins[base + i] + code[base + i] as f32 * steps[base + i]);
            sum += d * d;
        }
        sum
    }

    /// Walk the HNSW graph using SQ8 approximate distance.
    fn search_quantized(&self, query: &[f32], ef: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let codes = &self.codes;
        let dim = self.index.dimension;
        let mins = &self.mins;
        let steps = &self.steps;

        let (entry_point, entry_layer) = self.index.entry_point().unwrap_or((0, 0));

        // Navigate upper layers with greedy single-node descent.
        let mut current = entry_point;
        let code_slice = &codes[current as usize * dim..(current as usize + 1) * dim];
        let mut current_dist = Self::approx_dist(query, code_slice, mins, steps);

        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.index.layers.len() {
                continue;
            }
            let layer = &self.index.layers[layer_idx];
            let mut changed = true;
            while changed {
                changed = false;
                for &neighbor_id in layer.get_neighbors(current).iter() {
                    let ncode =
                        &codes[neighbor_id as usize * dim..(neighbor_id as usize + 1) * dim];
                    let dist = Self::approx_dist(query, ncode, mins, steps);
                    if dist < current_dist {
                        current_dist = dist;
                        current = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Base layer: beam search with SQ8 approximate distance + prefetch.
        if self.index.layers.is_empty() {
            return Ok(Vec::new());
        }
        let base_layer = &self.index.layers[0];
        let dist_fn = |_q: &[f32], node_id: u32| -> f32 {
            // Prefetch next cache line of codes (hide L2 latency).
            let offset = node_id as usize * dim;
            let ncode = &codes[offset..offset + dim];
            Self::approx_dist(query, ncode, mins, steps)
        };
        Ok(crate::hnsw::search::greedy_search_layer_custom(
            query,
            current,
            base_layer,
            &self.index.vectors,
            self.index.dimension,
            ef,
            &dist_fn,
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    fn random_normalized(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
                crate::distance::normalize(&v)
            })
            .collect()
    }

    fn l2_params() -> crate::hnsw::HNSWParams {
        crate::hnsw::HNSWParams {
            metric: crate::distance::DistanceMetric::L2,
            ..Default::default()
        }
    }

    #[test]
    fn sq8_encode_decode_roundtrip() {
        let dim = 128;
        let n = 100;
        let vecs = random_vectors(n, dim, 42);

        let mut index = HNSWSq8Index::with_params(dim, l2_params()).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Check that decoded values are within 1/255 of the range.
        let vectors = index.index.raw_vectors();
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let c = &index.codes[i * dim..(i + 1) * dim];
            for d in 0..dim {
                let decoded = index.mins[d] + c[d] as f32 * index.steps[d];
                let max_err = index.steps[d]; // one step width
                assert!(
                    (decoded - v[d]).abs() <= max_err + 1e-6,
                    "vec {i} dim {d}: decoded={decoded} vs original={}, err={}, max_err={max_err}",
                    v[d],
                    (decoded - v[d]).abs(),
                );
            }
        }
    }

    #[test]
    fn sq8_self_retrieval() {
        let dim = 64;
        let n = 200;
        let vecs = random_vectors(n, dim, 99);

        let mut index = HNSWSq8Index::with_params(dim, l2_params()).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Each vector should retrieve itself as closest.
        let query = &vecs[0];
        let results = index.search(query, 5, 64).unwrap();
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "Query vector should be in its own top-5 SQ8 results"
        );
    }

    #[test]
    fn sq8_reranked_recall() {
        let dim = 32;
        let n = 500;
        let k = 10;
        let vecs = random_vectors(n, dim, 42);

        let mut index = HNSWSq8Index::with_params(dim, l2_params()).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Brute-force ground truth (L2).
        let query = &vecs[0];
        let mut gt: Vec<(u32, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, crate::distance::l2_distance(query, v)))
            .collect();
        gt.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_ids: std::collections::HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();

        let results = index.search_reranked(query, k, 64, 50).unwrap();
        let result_ids: std::collections::HashSet<u32> =
            results.iter().map(|(id, _)| *id).collect();

        let recall = gt_ids.intersection(&result_ids).count() as f32 / k as f32;
        assert!(
            recall >= 0.70,
            "SQ8 reranked recall@{k} = {recall:.3}, expected >= 0.70"
        );
    }

    #[test]
    fn sq8_with_l2_params() {
        use crate::distance::DistanceMetric;
        use crate::hnsw::HNSWParams;

        let dim = 32;
        let n = 200;
        let vecs = random_vectors(n, dim, 55);

        let params = HNSWParams {
            metric: DistanceMetric::L2,
            ..Default::default()
        };
        let mut index = HNSWSq8Index::with_params(dim, params).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let results = index.search_reranked(&vecs[0], 5, 64, 50).unwrap();
        // Self should be closest (distance ~0).
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-3, "self-distance should be near 0");
    }

    #[test]
    fn sq8_compression_ratio() {
        let dim = 128;
        let n = 1000;
        let vecs = random_vectors(n, dim, 42);

        let mut index = HNSWSq8Index::with_params(dim, l2_params()).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let float_bytes = n * dim * 4;
        let code_bytes = index.code_memory();
        let ratio = float_bytes as f64 / code_bytes as f64;

        // SQ8 = 1 byte/dim vs 4 bytes/dim = 4x compression.
        assert!(
            ratio > 3.8 && ratio < 4.2,
            "expected ~4x compression, got {ratio:.1}x"
        );
    }

    #[test]
    fn sq8_approx_dist_correlates_with_exact() {
        let dim = 128;
        let n = 100;
        let vecs = random_vectors(n, dim, 77);

        let mut index = HNSWSq8Index::with_params(dim, l2_params()).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query = &vecs[0];
        let vectors = index.index.raw_vectors();

        // Compute exact and approx distances, check rank correlation.
        let mut exact_dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                (i, crate::distance::l2_distance(query, v))
            })
            .collect();
        let mut approx_dists: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let code = &index.codes[i * dim..(i + 1) * dim];
                (
                    i,
                    HNSWSq8Index::approx_dist(query, code, &index.mins, &index.steps),
                )
            })
            .collect();

        exact_dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        approx_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Top-10 from exact should mostly appear in top-20 from approx.
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();
        let approx_top20: std::collections::HashSet<usize> =
            approx_dists.iter().take(20).map(|(i, _)| *i).collect();

        let overlap = exact_top10.intersection(&approx_top20).count();
        assert!(
            overlap >= 8,
            "SQ8 approx should preserve ranking: {overlap}/10 of exact top-10 in approx top-20"
        );
    }

    #[test]
    fn sq8_cosine_metric() {
        let dim = 32;
        let n = 200;
        let vecs = random_normalized(n, dim, 42);

        // Default metric is cosine.
        let mut index = HNSWSq8Index::new(dim, 16, 32).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let results = index.search_reranked(&vecs[0], 5, 64, 50).unwrap();
        assert_eq!(results[0].0, 0, "self should be closest");
    }
}
