//! SymphonyQG: HNSW with RaBitQ quantized graph traversal.
//!
//! Co-locates RaBitQ quantized codes alongside the HNSW graph so beam search
//! uses approximate distance (cheap) instead of full-precision f32 distance.
//! The query rotation is precomputed once per search; per-neighbor distance is
//! a single O(d) dot product over u16 codes.
//!
//! # Two-stage search
//!
//! 1. Graph traversal with RaBitQ approximate L2 distance (no raw vector access)
//! 2. Optional reranking of top candidates with exact f32 distance
//!
//! # Example
//!
//! ```rust,no_run
//! # fn main() -> Result<(), vicinity::RetrieveError> {
//! use vicinity::hnsw::symphony_qg::SymphonyQGIndex;
//!
//! let dim = 128;
//! let mut index = SymphonyQGIndex::new(dim, 16, 16)?;
//!
//! let v = vicinity::distance::normalize(&vec![0.1; dim]);
//! index.add_slice(0, &v)?;
//! // ... add more vectors ...
//!
//! // Build HNSW graph then quantize
//! index.build()?;
//!
//! // Search with quantized graph traversal + exact reranking
//! let q = vicinity::distance::normalize(&vec![0.15; dim]);
//! let results = index.search_reranked(&q, 10, 50, 100)?;
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - Gou et al. (2025). "SymphonyQG: Towards Symphonious Integration of
//!   Quantization and Graph for ANN Search." SIGMOD 2025.

use crate::hnsw::graph::HNSWIndex;
use crate::RetrieveError;
use qntz::rabitq::{QuantizedVector, RaBitQConfig, RaBitQQuantizer};

/// HNSW index with RaBitQ quantized graph traversal.
///
/// Graph construction uses full-precision f32 vectors (for quality). Search
/// walks the HNSW graph using pre-rotated RaBitQ approximate distances: the
/// query is rotated once, then each neighbor's distance is a single O(d)
/// dot product over quantized codes.
///
/// Memory: stores both f32 vectors (for reranking) and quantized codes
/// (~2 bytes/dim for 4-bit RaBitQ) alongside the graph.
pub struct SymphonyQGIndex {
    /// The underlying HNSW index (owns graph + f32 vectors).
    index: HNSWIndex,
    /// Per-vector quantized codes, indexed by internal id.
    codes: Vec<QuantizedVector>,
    /// RaBitQ quantizer (owns rotation matrix and centroid for pre-rotation).
    quantizer: Option<RaBitQQuantizer>,
    /// RaBitQ configuration.
    rabitq_config: RaBitQConfig,
    /// Random seed for rotation matrix.
    seed: u64,
    /// Whether quantization has been performed.
    quantized_built: bool,
}

impl SymphonyQGIndex {
    /// Create a new SymphonyQG index with 4-bit RaBitQ (default).
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        Self::with_config(dimension, m, m_max, RaBitQConfig::bits4(), 42)
    }

    /// Create with specific RaBitQ configuration.
    pub fn with_config(
        dimension: usize,
        m: usize,
        m_max: usize,
        rabitq_config: RaBitQConfig,
        seed: u64,
    ) -> Result<Self, RetrieveError> {
        let index = HNSWIndex::new(dimension, m, m_max)?;
        Ok(Self {
            index,
            codes: Vec::new(),
            quantizer: None,
            rabitq_config,
            seed,
            quantized_built: false,
        })
    }

    /// Create with full HNSW params and RaBitQ config.
    ///
    /// Use this when the default cosine metric is wrong (e.g., L2 distance datasets).
    pub fn with_hnsw_params(
        dimension: usize,
        params: super::graph::HNSWParams,
        rabitq_config: RaBitQConfig,
        seed: u64,
    ) -> Result<Self, RetrieveError> {
        let index = HNSWIndex::with_params(dimension, params)?;
        Ok(Self {
            index,
            codes: Vec::new(),
            quantizer: None,
            rabitq_config,
            seed,
            quantized_built: false,
        })
    }

    /// Add a vector. Must be L2-normalized for cosine distance.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.index.add_slice(doc_id, vector)
    }

    /// Build the HNSW graph (f32) and then quantize all vectors.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        self.index.build()?;
        self.quantize_vectors()?;
        Ok(())
    }

    /// Quantize all vectors using RaBitQ.
    fn quantize_vectors(&mut self) -> Result<(), RetrieveError> {
        let n = self.index.num_vectors;
        if n == 0 {
            self.quantized_built = true;
            return Ok(());
        }
        let dim = self.index.dimension;

        // Create quantizer and fit centroid from data.
        let mut quantizer = RaBitQQuantizer::with_config(dim, self.seed, self.rabitq_config)
            .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ init: {e}")))?;
        quantizer
            .fit(&self.index.vectors, n)
            .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ fit: {e}")))?;

        // Quantize each vector.
        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let vec = self.index.get_vector(i);
            let qv = quantizer
                .quantize(vec)
                .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ quantize: {e}")))?;
            codes.push(qv);
        }

        self.quantizer = Some(quantizer);
        self.codes = codes;
        self.quantized_built = true;
        Ok(())
    }

    /// Search using quantized graph traversal (no reranking).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.check_search_ready(query)?;

        let results = self.search_quantized_graph(query, ef)?;
        let mut output: Vec<(u32, f32)> = results
            .into_iter()
            .take(k)
            .map(|(internal_id, dist)| (self.index.doc_ids[internal_id as usize], dist))
            .collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output)
    }

    /// Search with oversampling + exact f32 reranking.
    ///
    /// 1. Retrieve `rerank_pool` candidates using quantized graph traversal
    /// 2. Rerank using exact f32 cosine distance
    /// 3. Return top `k`
    pub fn search_reranked(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank_pool: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.check_search_ready(query)?;

        let pool = rerank_pool.max(k);
        let candidates = self.search_quantized_graph(query, ef.max(pool))?;

        let dist_fn = self.index.dist_fn();
        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(pool)
            .map(|(internal_id, _approx_dist)| {
                let vec = self.index.get_vector(internal_id as usize);
                let exact_dist = dist_fn(query, vec);
                (self.index.doc_ids[internal_id as usize], exact_dist)
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

    // ── internal ──────────────────────────────────────────────────────────

    fn check_search_ready(&self, query: &[f32]) -> Result<(), RetrieveError> {
        if !self.index.is_built() {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if !self.quantized_built {
            return Err(RetrieveError::InvalidParameter(
                "quantization not built (call build())".into(),
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

    /// Pre-rotate query: subtract centroid, apply rotation matrix.
    /// O(d^2) -- called once per query, amortized across all neighbor evaluations.
    fn rotate_query(&self, query: &[f32]) -> Result<Vec<f32>, RetrieveError> {
        self.quantizer
            .as_ref()
            .ok_or_else(|| {
                RetrieveError::InvalidParameter("quantizer must be set after build".into())
            })?
            .rotate_query(query)
            .map_err(|e| RetrieveError::InvalidParameter(format!("rotate query: {e}")))
    }

    /// Walk the HNSW graph using RaBitQ approximate distance.
    ///
    /// Upper layers: greedy single-node descent with quantized distance.
    /// Base layer: delegates to `greedy_search_layer_custom` with a closure
    /// that computes approximate L2 from the pre-rotated query.
    fn search_quantized_graph(
        &self,
        query: &[f32],
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let rotated_query = self.rotate_query(query)?;
        let codes = &self.codes;

        // Use cached entry point (O(1) vs O(n) scan).
        let (entry_point, entry_layer) = self.index.entry_point().unwrap_or((0, 0));

        // Navigate upper layers with greedy single-node descent.
        let mut current = entry_point;
        let mut current_dist = approx_dist_sqr(&rotated_query, &codes[current as usize]);

        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.index.layers.len() {
                continue;
            }
            let layer = &self.index.layers[layer_idx];
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = layer.get_neighbors(current);
                for &neighbor_id in neighbors.iter() {
                    let dist = approx_dist_sqr(&rotated_query, &codes[neighbor_id as usize]);
                    if dist < current_dist {
                        current_dist = dist;
                        current = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Base layer: use the shared beam search with custom distance.
        if self.index.layers.is_empty() {
            return Ok(Vec::new());
        }
        let base_layer = &self.index.layers[0];
        let dist_fn = |_q: &[f32], node_id: u32| -> f32 {
            approx_dist_sqr(&rotated_query, &codes[node_id as usize])
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

/// Approximate L2 squared distance from a pre-rotated query to a quantized vector.
/// Returns L2^2 (no sqrt) -- monotonic with L2, correct for ranking.
#[inline]
fn approx_dist_sqr(rotated_query: &[f32], qv: &QuantizedVector) -> f32 {
    RaBitQQuantizer::approximate_l2_sqr_prerotated(rotated_query, qv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normalized_vector(seed: usize, dim: usize) -> Vec<f32> {
        let v: Vec<f32> = (0..dim)
            .map(|j| ((seed * dim + j) as f32 * 0.618_034).fract() * 2.0 - 1.0)
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_symphony_qg_basic() {
        let dim = 32;
        let n = 200;
        let mut index = SymphonyQGIndex::new(dim, 8, 8).unwrap();

        for i in 0..n {
            index
                .add_slice(i as u32, &make_normalized_vector(i, dim))
                .unwrap();
        }
        index.build().unwrap();

        let q = make_normalized_vector(0, dim);
        let results = index.search_reranked(&q, 5, 32, 50).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0, "self-query should return doc_id 0");
    }

    #[test]
    fn test_distance_matches_qntz() {
        // Verify prerotated distance matches qntz's approximate_l2_sqr
        let dim = 32;
        let n = 50;
        let seed = 42;
        let config = RaBitQConfig::bits4();

        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vector(i, dim)).collect();
        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let mut quantizer = RaBitQQuantizer::with_config(dim, seed, config).unwrap();
        quantizer.fit(&flat, n).unwrap();

        let codes: Vec<QuantizedVector> = vectors
            .iter()
            .map(|v| quantizer.quantize(v).unwrap())
            .collect();

        let query = &vectors[0];

        // qntz standard distance (rotates internally each call)
        let qntz_dist = quantizer.approximate_l2_sqr(query, &codes[1]).unwrap();

        // prerotated API distance
        let rotated = quantizer.rotate_query(query).unwrap();
        let prerotated_dist = RaBitQQuantizer::approximate_l2_sqr_prerotated(&rotated, &codes[1]);

        let diff = (qntz_dist - prerotated_dist).abs();
        assert!(
            diff < 1e-4,
            "distance mismatch: qntz={qntz_dist}, prerotated={prerotated_dist}, diff={diff}"
        );
    }

    #[test]
    fn test_symphony_qg_recall() {
        // RaBitQ approximation quality improves with dimension (O(1/sqrt(d))).
        // Use dim=256 and generous ef/rerank for reliable recall.
        let dim = 256;
        let n = 300;
        let mut index =
            SymphonyQGIndex::with_config(dim, 16, 16, RaBitQConfig::bits4(), 42).unwrap();

        let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_normalized_vector(i, dim)).collect();
        for (i, v) in vectors.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Reranked search: quantized traversal finds candidates, exact f32 reranks.
        let mut hits = 0;
        for (i, v) in vectors.iter().enumerate() {
            let results = index.search_reranked(v, 1, 200, 100).unwrap();
            if results.first().map(|(id, _)| *id) == Some(i as u32) {
                hits += 1;
            }
        }
        let recall = hits as f64 / n as f64;
        assert!(
            recall > 0.5,
            "reranked self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    /// Diagnostic: check if RaBitQ approximate distance is correlated with true L2
    /// on unnormalized vectors (varying norms).
    #[test]
    fn test_rabitq_distance_correlation_unnormalized() {
        let dim = 128;
        let n = 100;

        // Generate vectors with norm ~5-10 (unnormalized, like GIST)
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|seed| {
                let v: Vec<f32> = (0..dim)
                    .map(|j| ((seed * dim + j) as f32 * 0.618_034).fract() * 2.0 - 1.0)
                    .collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let target_norm = 5.0 + (seed as f32 % 5.0); // norms from 5 to 10
                v.iter().map(|x| x * target_norm / norm).collect()
            })
            .collect();

        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let mut quantizer = RaBitQQuantizer::with_config(dim, 42, RaBitQConfig::bits4()).unwrap();
        quantizer.fit(&flat, n).unwrap();

        let codes: Vec<QuantizedVector> = vectors
            .iter()
            .map(|v| quantizer.quantize(v).unwrap())
            .collect();

        // Check: for a query, does the approximate ranking correlate with true L2?
        let query = &vectors[0];
        let rotated = quantizer.rotate_query(query).unwrap();

        let mut true_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .skip(1) // skip self
            .map(|(i, v)| {
                let d: f32 = query
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                (i, d)
            })
            .collect();
        true_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut approx_dists: Vec<(usize, f32)> = (1..n)
            .map(|i| {
                let d = RaBitQQuantizer::approximate_l2_sqr_prerotated(&rotated, &codes[i]);
                (i, d)
            })
            .collect();
        approx_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Check recall@10: how many of true top-10 are in approx top-10?
        let true_top10: std::collections::HashSet<usize> =
            true_dists.iter().take(10).map(|(i, _)| *i).collect();
        let approx_top10: std::collections::HashSet<usize> =
            approx_dists.iter().take(10).map(|(i, _)| *i).collect();
        let overlap = true_top10.intersection(&approx_top10).count();

        eprintln!(
            "RaBitQ unnormalized recall@10: {}/10 (true top-10 vs approx top-10)",
            overlap
        );
        eprintln!(
            "  f_add range: {:.1}..{:.1}",
            codes.iter().map(|c| c.f_add).fold(f32::INFINITY, f32::min),
            codes
                .iter()
                .map(|c| c.f_add)
                .fold(f32::NEG_INFINITY, f32::max),
        );
        eprintln!(
            "  f_rescale range: {:.4}..{:.4}",
            codes
                .iter()
                .map(|c| c.f_rescale)
                .fold(f32::INFINITY, f32::min),
            codes
                .iter()
                .map(|c| c.f_rescale)
                .fold(f32::NEG_INFINITY, f32::max),
        );
        eprintln!(
            "  residual_norm range: {:.2}..{:.2}",
            codes
                .iter()
                .map(|c| c.residual_norm)
                .fold(f32::INFINITY, f32::min),
            codes
                .iter()
                .map(|c| c.residual_norm)
                .fold(f32::NEG_INFINITY, f32::max),
        );

        // With badly varying norms, we expect low recall
        // This test documents the current behavior -- it's a diagnostic, not an assertion
        if overlap <= 2 {
            eprintln!("WARNING: RaBitQ distance approximation is broken for unnormalized vectors");
            eprintln!("  The correction factors (f_add, f_rescale) scale with ||residual||^2,");
            eprintln!(
                "  which varies wildly for unnormalized data, drowning the discriminative IP."
            );
        }
        // For now, we just document: assert it's at least not zero
        assert!(
            overlap >= 1 || n < 20,
            "RaBitQ has zero correlation with true L2 on unnormalized data"
        );
    }

    /// End-to-end test: SymphonyQG with L2 metric on unnormalized vectors.
    #[test]
    fn test_symphony_qg_l2_unnormalized() {
        use crate::distance::DistanceMetric;
        use crate::hnsw::graph::HNSWParams;

        let dim = 64;
        let n = 200;

        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|seed| {
                let v: Vec<f32> = (0..dim)
                    .map(|j| ((seed * dim + j) as f32 * 0.618_034).fract() * 2.0 - 1.0)
                    .collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                let target_norm = 5.0 + (seed as f32 % 5.0);
                v.iter().map(|x| x * target_norm / norm).collect()
            })
            .collect();

        let params = HNSWParams {
            m: 16,
            m_max: 32,
            ef_construction: 200,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index =
            SymphonyQGIndex::with_hnsw_params(dim, params, RaBitQConfig::bits4(), 42).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Raw quantized search
        let q = &vectors[0];
        let raw_results = index.search(q, 10, 100).unwrap();
        eprintln!(
            "L2 raw quantized: {} results, top={:?}",
            raw_results.len(),
            raw_results.first()
        );

        // Reranked search
        let reranked_results = index.search_reranked(q, 10, 100, 50).unwrap();
        eprintln!(
            "L2 reranked: {} results, top={:?}",
            reranked_results.len(),
            reranked_results.first()
        );

        // Self-query should return self as nearest
        assert!(!raw_results.is_empty(), "raw search returned no results");

        // Check that reranked actually uses L2, not cosine
        // The nearest result for self-query should be doc_id 0
        // With cosine reranking on unnormalized vectors, this may fail
        assert_eq!(
            reranked_results[0].0, 0,
            "self-query should return doc_id 0 (got {}), \
             likely rerank uses wrong distance metric",
            reranked_results[0].0
        );
    }
}
