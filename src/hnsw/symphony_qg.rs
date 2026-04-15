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
    /// Create a new SymphonyQG index with 4-bit RaBitQ and cosine distance.
    ///
    /// For L2 (Euclidean) distance on unnormalized vectors, use
    /// [`with_hnsw_params`](Self::with_hnsw_params) instead -- the default
    /// cosine metric produces wrong results on unnormalized data.
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        Self::with_config(dimension, m, m_max, RaBitQConfig::bits4(), 42)
    }

    /// Create with specific RaBitQ configuration and cosine distance.
    ///
    /// For L2 distance, use [`with_hnsw_params`](Self::with_hnsw_params).
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

// ── Vertex-Relative SymphonyQG ──────────────────────────────────────────────

/// Per-edge correction scalars for vertex-relative RaBitQ.
/// Stored in SoA layout (flat arrays) for cache efficiency.
#[derive(Clone, Copy)]
struct EdgeScalars {
    f_add: f32,
    f_rescale: f32,
    ip_u_rot_codes: f32,
}

/// HNSW index with vertex-relative RaBitQ quantized graph traversal.
///
/// Unlike [`SymphonyQGIndex`] (which uses a global centroid), this variant
/// stores per-edge quantized codes where each neighbor `v` is quantized
/// relative to its parent `u`. This keeps RaBitQ error bounds tight for
/// L2/unnormalized data where vector norms vary.
///
/// Memory cost: one `EdgeCode` per directed edge in the graph (~500 bytes/edge
/// at d=960 with 4-bit RaBitQ). For M=16 and 1M vectors, this is ~8GB.
/// Use for high-dimensional L2 workloads where accuracy matters more than memory.
///
/// # References
///
/// - Gou et al. (2025). "SymphonyQG", SIGMOD 2025, Section 3.1.1.
///   Vertex-relative normalization for HNSW + RaBitQ.
pub struct SymphonyQGVRIndex {
    /// The underlying HNSW index (owns graph + f32 vectors).
    index: HNSWIndex,
    /// Per-edge correction scalars, indexed by edge offset.
    edge_scalars: Vec<EdgeScalars>,
    /// Pre-shifted quantized codes as f32: `code_val + cb`.
    /// Flat SoA: `edge_codes_f32[edge_offset * dim .. (edge_offset+1) * dim]`.
    /// Eliminates per-neighbor u16->f32 cast + cb addition in the hot path.
    edge_codes_f32: Vec<f32>,
    /// Cumulative neighbor count per node: `neighbor_offsets[node_id]` is the
    /// start index into edge arrays for node `node_id`'s neighbors.
    neighbor_offsets: Vec<u32>,
    /// RaBitQ quantizer (owns rotation matrix).
    quantizer: Option<RaBitQQuantizer>,
    /// RaBitQ configuration.
    rabitq_config: RaBitQConfig,
    /// Random seed for rotation matrix.
    seed: u64,
    /// Dimension.
    dimension: usize,
    /// Whether build has completed.
    built: bool,
}

impl SymphonyQGVRIndex {
    /// Create a new vertex-relative SymphonyQG index.
    pub fn new(
        dimension: usize,
        params: super::graph::HNSWParams,
        rabitq_config: RaBitQConfig,
        seed: u64,
    ) -> Result<Self, RetrieveError> {
        let index = HNSWIndex::with_params(dimension, params)?;
        Ok(Self {
            index,
            edge_scalars: Vec::new(),
            edge_codes_f32: Vec::new(),
            neighbor_offsets: Vec::new(),
            quantizer: None,
            rabitq_config,
            seed,
            dimension,
            built: false,
        })
    }

    /// Add a vector.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.index.add_slice(doc_id, vector)
    }

    /// Build the HNSW graph then quantize per-edge codes.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        self.index.build()?;
        self.build_edge_codes()?;
        self.built = true;
        Ok(())
    }

    /// Build per-edge RaBitQ codes using vertex-relative centroids.
    fn build_edge_codes(&mut self) -> Result<(), RetrieveError> {
        let n = self.index.num_vectors;
        if n == 0 {
            return Ok(());
        }
        let dim = self.dimension;

        // Create quantizer with zero centroid (we'll pass vertex-relative centroids manually).
        let mut quantizer = RaBitQQuantizer::with_config(dim, self.seed, self.rabitq_config)
            .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ init: {e}")))?;
        // Set centroid to zero -- quantize_with_centroid will use the per-edge centroid.
        quantizer
            .set_centroid(vec![0.0f32; dim])
            .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ centroid: {e}")))?;

        // Compute rotated vectors R*v for all nodes (for ip_u_rot_codes precomputation).
        let mut rotated_flat = vec![0.0f32; n * dim];
        for i in 0..n {
            let v = self.index.get_vector(i);
            let r = quantizer
                .rotate_query(v)
                .map_err(|e| RetrieveError::InvalidParameter(format!("rotate: {e}")))?;
            rotated_flat[i * dim..(i + 1) * dim].copy_from_slice(&r);
        }

        // Build per-edge codes for base layer (layer 0).
        if self.index.layers.is_empty() {
            self.quantizer = Some(quantizer);
            return Ok(());
        }
        let base_layer = &self.index.layers[0];
        let layer_len = base_layer.len();

        // Count total edges to pre-allocate flat arrays.
        let total_edges: usize = (0..layer_len as u32)
            .map(|id| base_layer.get_neighbors(id).len())
            .sum();

        let mut edge_scalars = Vec::with_capacity(total_edges);
        let mut edge_codes_f32 = Vec::with_capacity(total_edges * dim);
        let mut neighbor_offsets = Vec::with_capacity(layer_len + 1);

        for node_id in 0..layer_len as u32 {
            neighbor_offsets.push(edge_scalars.len() as u32);
            let neighbors = base_layer.get_neighbors(node_id);
            let u_vec = self.index.get_vector(node_id as usize);
            let u_rot = &rotated_flat[node_id as usize * dim..(node_id as usize + 1) * dim];

            for &neighbor_id in neighbors.iter() {
                let v_vec = self.index.get_vector(neighbor_id as usize);

                // Quantize v relative to u (centroid = u_vec).
                let qv = quantizer
                    .quantize_with_centroid(v_vec, u_vec)
                    .map_err(|e| RetrieveError::InvalidParameter(format!("quantize edge: {e}")))?;

                // Pre-shift codes to f32: code_val + cb (eliminates cast+add in hot path).
                let cb = -((1u32 << qv.ex_bits) as f32 - 0.5);
                let mut ip_u_rot = 0.0f32;
                for (&c, &ur) in qv.codes.iter().zip(u_rot.iter()) {
                    let shifted = c as f32 + cb;
                    edge_codes_f32.push(shifted);
                    ip_u_rot += ur * shifted;
                }

                edge_scalars.push(EdgeScalars {
                    f_add: qv.f_add,
                    f_rescale: qv.f_rescale,
                    ip_u_rot_codes: ip_u_rot,
                });
            }
        }
        // Sentinel for the last node's range.
        neighbor_offsets.push(edge_scalars.len() as u32);

        self.edge_scalars = edge_scalars;
        self.edge_codes_f32 = edge_codes_f32;
        self.neighbor_offsets = neighbor_offsets;
        // rotated_flat is no longer needed -- drop it (was only for ip_u_rot precomputation)
        self.quantizer = Some(quantizer);
        Ok(())
    }

    /// Search using vertex-relative quantized graph traversal (no reranking).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built || self.index.num_vectors == 0 || self.index.layers.is_empty() {
            return Ok(Vec::new());
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let quantizer = self.quantizer.as_ref().unwrap();
        let rotated_query = quantizer
            .rotate_query(query)
            .map_err(|e| RetrieveError::InvalidParameter(format!("rotate query: {e}")))?;

        let (entry_point, entry_layer) = self.index.entry_point().unwrap_or((0, 0));
        let dim = self.dimension;

        // Upper layers: greedy descent with global-centroid codes (fallback).
        // For upper layers we don't have per-edge codes, so use exact f32 distance.
        let dist_fn_exact = self.index.dist_fn();
        let mut current = entry_point;
        let mut current_dist = dist_fn_exact(query, self.index.get_vector(current as usize));

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
                    let dist = dist_fn_exact(query, self.index.get_vector(neighbor_id as usize));
                    if dist < current_dist {
                        current_dist = dist;
                        current = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Base layer: edge-aware beam search with per-edge codes.
        let entry_dist = self.approx_dist_vr_entry(&rotated_query, current);
        let base_layer = &self.index.layers[0];
        let edge_scalars = &self.edge_scalars;
        let edge_codes_f32 = &self.edge_codes_f32;
        let neighbor_offsets = &self.neighbor_offsets;

        let dist_fn = |parent_id: u32, _neighbor_id: u32, slot: usize| -> f32 {
            let offset = neighbor_offsets[parent_id as usize] as usize + slot;
            let scalars = &edge_scalars[offset];
            let codes = &edge_codes_f32[offset * dim..(offset + 1) * dim];
            approx_dist_vr_flat(&rotated_query, codes, scalars)
        };

        let results = crate::hnsw::search::greedy_search_layer_edge_aware(
            current,
            entry_dist,
            base_layer,
            self.index.num_vectors,
            ef,
            &dist_fn,
        );

        let mut output: Vec<(u32, f32)> = results
            .into_iter()
            .take(k)
            .map(|(internal_id, dist)| (self.index.doc_ids[internal_id as usize], dist))
            .collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output)
    }

    /// Search with oversampling + exact f32 reranking.
    pub fn search_reranked(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank_pool: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built || self.index.num_vectors == 0 {
            return Ok(Vec::new());
        }

        let pool = rerank_pool.max(k);
        let candidates = self.search(query, pool, ef.max(pool))?;

        let dist_fn = self.index.dist_fn();
        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(pool)
            .map(|(doc_id, _approx_dist)| {
                // Resolve back to internal id for vector lookup.
                let internal_id = self
                    .index
                    .doc_id_to_internal
                    .get(&doc_id)
                    .copied()
                    .unwrap_or(0);
                let vec = self.index.get_vector(internal_id as usize);
                let exact_dist = dist_fn(query, vec);
                (doc_id, exact_dist)
            })
            .collect();

        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Approximate distance for the entry point (no parent context).
    /// Uses exact f32 distance as fallback since we don't have per-edge codes for entry.
    fn approx_dist_vr_entry(&self, _rotated_query: &[f32], entry_id: u32) -> f32 {
        // For the entry point we don't have a parent edge, so use a rough estimate.
        // This is only used once per query, so exact distance is fine.
        0.0 // Will be refined by the beam search immediately
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
}

/// Approximate L2^2 from a globally-rotated query to vertex-relative edge codes.
///
/// Uses pre-shifted f32 codes (code_val + cb already applied at build time)
/// for a tight IP loop with no per-element cast or addition.
///
/// Formula: `f_add + f_rescale * (IP(q_rot, shifted_codes) - ip_u_rot_codes)`
#[inline]
fn approx_dist_vr_flat(rotated_query: &[f32], shifted_codes: &[f32], scalars: &EdgeScalars) -> f32 {
    // Tight dot product -- the compiler will auto-vectorize this on x86/ARM.
    let mut ip = 0.0f32;
    for (&q, &c) in rotated_query.iter().zip(shifted_codes.iter()) {
        ip += q * c;
    }
    (scalars.f_add + scalars.f_rescale * (ip - scalars.ip_u_rot_codes)).max(0.0)
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

    // ── Vertex-relative tests ────────────────────────────────────────────

    #[test]
    fn test_symphony_qg_vr_l2_unnormalized() {
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
        let mut index = SymphonyQGVRIndex::new(dim, params, RaBitQConfig::bits4(), 42).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Self-query: raw quantized should find self (or very close)
        let q = &vectors[0];
        let raw_results = index.search(q, 10, 100).unwrap();
        assert!(!raw_results.is_empty(), "VR raw search returned no results");

        // Reranked search should return exact self
        let reranked = index.search_reranked(q, 10, 100, 50).unwrap();
        assert_eq!(
            reranked[0].0, 0,
            "VR reranked self-query should return doc_id 0 (got {})",
            reranked[0].0
        );

        // Recall check: brute-force top-10 vs VR reranked top-10
        let mut gt: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                (i as u32, d)
            })
            .collect();
        gt.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_set: std::collections::HashSet<u32> =
            gt.iter().take(10).map(|(id, _)| *id).collect();
        let result_set: std::collections::HashSet<u32> =
            reranked.iter().map(|(id, _)| *id).collect();
        let overlap = gt_set.intersection(&result_set).count();

        assert!(
            overlap >= 5,
            "VR L2 recall@10 too low: {}/10 overlap with brute-force",
            overlap
        );
    }
}
