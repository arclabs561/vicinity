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
/// Memory cost: `d/2` bytes per edge (packed 4-bit codes) + 12 bytes per edge (scalars).
/// At d=128, M=16, 1M vectors: ~1.5GB. At d=960, M=16, 1M vectors: ~10GB.
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
    /// Packed quantized codes. Packing depends on `total_bits`:
    /// - 4-bit: 2 codes per byte (nibbles). `packed_dim = ceil(d/2)`.
    /// - 1-bit: 8 codes per byte (binary). `packed_dim = ceil(d/8)`.
    packed_codes: Vec<u8>,
    /// Bytes per edge in packed_codes.
    packed_dim: usize,
    /// Total bits per code dimension (from RaBitQConfig).
    total_bits: usize,
    /// Code bias: `cb = -((1 << ex_bits) as f32 - 0.5)`. Constant for the index.
    cb: f32,
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
        let total_bits = rabitq_config.total_bits;
        let ex_bits = total_bits.saturating_sub(1);
        let cb = -((1u32 << ex_bits) as f32 - 0.5);
        // Packing: codes_per_byte = 8 / total_bits. For binary: 8. For 4-bit: 2.
        let codes_per_byte = 8 / total_bits.max(1);
        let packed_dim = (dimension + codes_per_byte - 1) / codes_per_byte;
        Ok(Self {
            index,
            edge_scalars: Vec::new(),
            packed_codes: Vec::new(),
            packed_dim,
            total_bits,
            cb,
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
        // Parallel: each rotation is O(d^2) and independent.
        let rotated_flat = {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                let vectors = &self.index.vectors;
                let mut flat = vec![0.0f32; n * dim];
                flat.par_chunks_mut(dim)
                    .enumerate()
                    .try_for_each(|(i, chunk)| {
                        let v = &vectors[i * dim..(i + 1) * dim];
                        let r = quantizer
                            .rotate_query(v)
                            .map_err(|e| RetrieveError::InvalidParameter(format!("rotate: {e}")))?;
                        chunk.copy_from_slice(&r);
                        Ok::<_, RetrieveError>(())
                    })?;
                flat
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut flat = vec![0.0f32; n * dim];
                for i in 0..n {
                    let v = self.index.get_vector(i);
                    let r = quantizer
                        .rotate_query(v)
                        .map_err(|e| RetrieveError::InvalidParameter(format!("rotate: {e}")))?;
                    flat[i * dim..(i + 1) * dim].copy_from_slice(&r);
                }
                flat
            }
        };

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

        let packed_dim = self.packed_dim;
        let total_bits = self.total_bits;
        let cb = self.cb;

        // Per-node edge quantization. Each node's edges are independent.
        // Collect (scalars, packed_bytes) per node, then flatten.
        let quantize_node = |node_id: u32| -> Result<(Vec<EdgeScalars>, Vec<u8>), RetrieveError> {
            let neighbors = base_layer.get_neighbors(node_id);
            let u_vec = self.index.get_vector(node_id as usize);
            let u_rot = &rotated_flat[node_id as usize * dim..(node_id as usize + 1) * dim];

            let mut scalars = Vec::with_capacity(neighbors.len());
            let mut codes = Vec::with_capacity(neighbors.len() * packed_dim);

            for &neighbor_id in neighbors.iter() {
                let v_rot =
                    &rotated_flat[neighbor_id as usize * dim..(neighbor_id as usize + 1) * dim];
                // Pre-rotated residual: R*(v - u) = R*v - R*u. O(d) not O(d^2).
                let rotated_residual: Vec<f32> = v_rot
                    .iter()
                    .zip(u_rot.iter())
                    .map(|(&v, &u)| v - u)
                    .collect();
                let qv = quantizer
                    .quantize_prerotated(&rotated_residual, u_vec)
                    .map_err(|e| RetrieveError::InvalidParameter(format!("quantize edge: {e}")))?;

                let mut ip_u_rot = 0.0f32;
                for (&c, &ur) in qv.codes.iter().zip(u_rot.iter()) {
                    ip_u_rot += ur * (c as f32 + cb);
                }

                pack_codes(&qv.codes, total_bits, dim, &mut codes);

                scalars.push(EdgeScalars {
                    f_add: qv.f_add,
                    f_rescale: qv.f_rescale,
                    ip_u_rot_codes: ip_u_rot,
                });
            }
            Ok((scalars, codes))
        };

        // Parallel or sequential per-node quantization.
        let per_node: Vec<(Vec<EdgeScalars>, Vec<u8>)> = {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                (0..layer_len as u32)
                    .into_par_iter()
                    .map(quantize_node)
                    .collect::<Result<Vec<_>, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..layer_len as u32)
                    .map(quantize_node)
                    .collect::<Result<Vec<_>, _>>()?
            }
        };

        // Flatten into contiguous arrays with offset table.
        let mut edge_scalars = Vec::with_capacity(total_edges);
        let mut packed_codes = Vec::with_capacity(total_edges * packed_dim);
        let mut neighbor_offsets = Vec::with_capacity(layer_len + 1);

        for (scalars, codes) in per_node {
            neighbor_offsets.push(edge_scalars.len() as u32);
            edge_scalars.extend(scalars);
            packed_codes.extend(codes);
        }
        neighbor_offsets.push(edge_scalars.len() as u32);

        self.edge_scalars = edge_scalars;
        self.packed_codes = packed_codes;
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
        let internal = self.search_internal(query, k, ef)?;
        Ok(internal
            .into_iter()
            .map(|(id, d)| (self.index.doc_ids[id as usize], d))
            .collect())
    }

    /// Internal search returning (internal_id, approx_dist).
    fn search_internal(
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

        let quantizer = self.quantizer.as_ref().ok_or_else(|| {
            RetrieveError::InvalidParameter("quantizer not built (call build())".into())
        })?;
        let rotated_query = quantizer
            .rotate_query(query)
            .map_err(|e| RetrieveError::InvalidParameter(format!("rotate query: {e}")))?;

        let (entry_point, entry_layer) = self.index.entry_point().unwrap_or((0, 0));

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
        let packed = &self.packed_codes;
        let neighbor_offsets = &self.neighbor_offsets;
        let packed_dim = self.packed_dim;
        // Precompute nibble LUT once per query (16 entries, 64 bytes, L1 cache).
        let lut = nibble_lut(self.cb);

        let total_bits = self.total_bits;
        let dist_fn = |parent_id: u32, _neighbor_id: u32, slot: usize| -> f32 {
            let base_offset = neighbor_offsets[parent_id as usize] as usize;
            let offset = base_offset + slot;

            // Prefetch next edge's codes to hide L2 cache latency.
            if offset + 2 < edge_scalars.len() {
                let ptr = packed.as_ptr().wrapping_add((offset + 2) * packed_dim);
                crate::hnsw::search::prefetch_read_data(ptr as *const f32);
            }

            let scalars = &edge_scalars[offset];
            let codes = &packed[offset * packed_dim..(offset + 1) * packed_dim];
            if total_bits >= 4 {
                approx_dist_vr_packed(&rotated_query, codes, scalars, &lut)
            } else {
                approx_dist_vr_binary(&rotated_query, codes, scalars)
            }
        };

        let results = crate::hnsw::search::greedy_search_layer_edge_aware(
            current,
            entry_dist,
            base_layer,
            self.index.num_vectors,
            ef,
            &dist_fn,
        );

        // Return internal IDs (not doc_ids) -- caller converts at the boundary.
        let mut output: Vec<(u32, f32)> = results.into_iter().take(k).collect();
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
        if self.is_compacted() {
            return Err(RetrieveError::InvalidParameter(
                "search_reranked unavailable after compact() -- use search() instead".into(),
            ));
        }

        let pool = rerank_pool.max(k);
        let candidates = self.search_internal(query, pool, ef.max(pool))?;

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

    /// Approximate distance for the entry point (no parent context).
    fn approx_dist_vr_entry(&self, _rotated_query: &[f32], _entry_id: u32) -> f32 {
        0.0 // Refined by the beam search immediately
    }

    /// Drop f32 vectors to reclaim memory. After compaction, `search_reranked`
    /// is unavailable -- only quantized `search` works.
    ///
    /// At d=960, 1M vectors, this saves 3.84 GB.
    pub fn compact(&mut self) {
        self.index.vectors.clear();
        self.index.vectors.shrink_to_fit();
    }

    /// Whether vectors have been compacted (reranking unavailable).
    pub fn is_compacted(&self) -> bool {
        self.index.vectors.is_empty() && self.index.num_vectors > 0
    }

    /// Search multiple queries in parallel.
    ///
    /// Returns one result set per query. Requires the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn search_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, RetrieveError> {
        use rayon::prelude::*;
        queries.par_iter().map(|q| self.search(q, k, ef)).collect()
    }

    /// Search + rerank multiple queries in parallel.
    #[cfg(feature = "parallel")]
    pub fn search_reranked_batch(
        &self,
        queries: &[&[f32]],
        k: usize,
        ef: usize,
        rerank_pool: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, RetrieveError> {
        use rayon::prelude::*;
        queries
            .par_iter()
            .map(|q| self.search_reranked(q, k, ef, rerank_pool))
            .collect()
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

    /// Memory usage breakdown in bytes.
    pub fn memory_usage_bytes(&self) -> VRMemoryReport {
        let vectors = self.index.vectors.len() * 4;
        let codes = self.packed_codes.len();
        let scalars = self.edge_scalars.len() * std::mem::size_of::<EdgeScalars>();
        let offsets = self.neighbor_offsets.len() * 4;
        let graph = self
            .index
            .layers
            .iter()
            .map(|l| l.len() * 16 * 4) // rough: nodes * avg_degree * u32
            .sum::<usize>();
        VRMemoryReport {
            vectors_bytes: vectors,
            packed_codes_bytes: codes,
            edge_scalars_bytes: scalars,
            graph_bytes: graph,
            offsets_bytes: offsets,
        }
    }
}

/// Memory breakdown for [`SymphonyQGVRIndex`].
pub struct VRMemoryReport {
    /// f32 vectors (for reranking).
    pub vectors_bytes: usize,
    /// Packed 4-bit edge codes.
    pub packed_codes_bytes: usize,
    /// Per-edge correction scalars.
    pub edge_scalars_bytes: usize,
    /// HNSW graph structure.
    pub graph_bytes: usize,
    /// Neighbor offset table.
    pub offsets_bytes: usize,
}

impl VRMemoryReport {
    /// Total bytes.
    pub fn total(&self) -> usize {
        self.vectors_bytes
            + self.packed_codes_bytes
            + self.edge_scalars_bytes
            + self.graph_bytes
            + self.offsets_bytes
    }
}

/// Pack quantized codes into bytes.
/// - 4-bit (total_bits=4): 2 codes per byte (high nibble, low nibble).
/// - 1-bit (total_bits=1): 8 codes per byte (MSB first).
#[inline]
fn pack_codes(codes: &[u16], total_bits: usize, dim: usize, out: &mut Vec<u8>) {
    if total_bits >= 4 {
        // 4-bit: two nibbles per byte
        let pairs = (dim + 1) / 2;
        for j in 0..pairs {
            let hi = (codes[j * 2] & 0x0F) as u8;
            let lo = if j * 2 + 1 < dim {
                (codes[j * 2 + 1] & 0x0F) as u8
            } else {
                0
            };
            out.push((hi << 4) | lo);
        }
    } else {
        // 1-bit: 8 bits per byte
        let bytes = (dim + 7) / 8;
        for j in 0..bytes {
            let mut byte = 0u8;
            for bit in 0..8 {
                let idx = j * 8 + bit;
                if idx < dim && codes[idx] != 0 {
                    byte |= 1 << (7 - bit);
                }
            }
            out.push(byte);
        }
    }
}

/// Precomputed nibble-to-f32 lookup table for 4-bit RaBitQ codes.
/// Maps each nibble value (0-15) to `nibble as f32 + cb`.
/// 16 entries * 4 bytes = 64 bytes -- fits in a cache line.
#[inline]
fn nibble_lut(cb: f32) -> [f32; 16] {
    let mut lut = [0.0f32; 16];
    for i in 0..16 {
        lut[i] = i as f32 + cb;
    }
    lut
}

/// Approximate L2^2 from a globally-rotated query to packed 4-bit edge codes.
///
/// Uses a 16-entry nibble LUT (64 bytes, fits in L1) to avoid per-element
/// u8->f32 conversion + cb addition. The hot loop is: load byte, two LUT
/// lookups, two FMAs.
///
/// Formula: `f_add + f_rescale * (IP(q_rot, codes + cb) - ip_u_rot_codes)`
#[inline]
fn approx_dist_vr_packed(
    rotated_query: &[f32],
    packed: &[u8],
    scalars: &EdgeScalars,
    lut: &[f32; 16],
) -> f32 {
    let mut ip = 0.0f32;
    let dim = rotated_query.len();
    let pairs = dim / 2;

    // Process two dimensions per byte -- the hot loop.
    // LUT lookups are branch-free and L1-resident.
    for j in 0..pairs {
        let byte = packed[j];
        let c0 = lut[(byte >> 4) as usize];
        let c1 = lut[(byte & 0x0F) as usize];
        ip += rotated_query[j * 2] * c0 + rotated_query[j * 2 + 1] * c1;
    }
    // Handle odd trailing dimension.
    if dim % 2 != 0 {
        let byte = packed[pairs];
        ip += rotated_query[dim - 1] * lut[(byte >> 4) as usize];
    }

    (scalars.f_add + scalars.f_rescale * (ip - scalars.ip_u_rot_codes)).max(0.0)
}

/// Approximate L2^2 using 1-bit (binary) packed codes.
///
/// Each byte holds 8 sign bits. Code value is 0 or 1, cb = -0.5.
/// The IP becomes: `sum(q[i] * (bit[i] - 0.5))` = `sum_positive - 0.5 * sum_all`
/// where `sum_positive` sums q[i] for bits that are 1.
#[inline]
fn approx_dist_vr_binary(rotated_query: &[f32], packed: &[u8], scalars: &EdgeScalars) -> f32 {
    let dim = rotated_query.len();
    let mut sum_positive = 0.0f32;
    let mut sum_all = 0.0f32;

    for j in 0..packed.len() {
        let byte = packed[j];
        for bit in 0..8 {
            let idx = j * 8 + bit;
            if idx >= dim {
                break;
            }
            let q = rotated_query[idx];
            sum_all += q;
            if byte & (1 << (7 - bit)) != 0 {
                sum_positive += q;
            }
        }
    }
    let ip = sum_positive - 0.5 * sum_all;
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

    /// Timing validation: measure build and search speed at moderate scale.
    /// This is a sanity check, not a performance regression test.
    #[test]
    fn test_vr_build_and_search_timing() {
        use crate::distance::DistanceMetric;
        use crate::hnsw::graph::HNSWParams;

        let dim = 128;
        let n = 5000;

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
            ef_construction: 100,
            metric: DistanceMetric::L2,
            seed: Some(42),
            ..Default::default()
        };
        let mut index = SymphonyQGVRIndex::new(dim, params, RaBitQConfig::bits4(), 42).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }

        // Time build
        let t0 = std::time::Instant::now();
        index.build().unwrap();
        let build_ms = t0.elapsed().as_millis();

        // Time search (100 queries)
        let queries: Vec<&[f32]> = vectors.iter().take(100).map(|v| v.as_slice()).collect();
        let t0 = std::time::Instant::now();
        for q in &queries {
            let _ = index.search(q, 10, 50);
        }
        let search_ms = t0.elapsed().as_millis();
        let qps = 100.0 / (search_ms as f64 / 1000.0);

        // Memory report
        let mem = index.memory_usage_bytes();

        eprintln!(
            "VR timing (n={n}, d={dim}): build={build_ms}ms, search={}ms ({qps:.0} QPS), \
             mem={:.1}MB (codes={:.1}MB, vectors={:.1}MB)",
            search_ms,
            mem.total() as f64 / 1e6,
            mem.packed_codes_bytes as f64 / 1e6,
            mem.vectors_bytes as f64 / 1e6,
        );

        // Sanity: build should complete in reasonable time
        assert!(
            build_ms < 60_000,
            "VR build took {build_ms}ms (>60s) for only {n} vectors at d={dim}"
        );
    }
}
