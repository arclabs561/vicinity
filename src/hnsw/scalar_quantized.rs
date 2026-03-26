//! HNSW with scalar quantization (SQ8).
//!
//! Wraps [`HNSWIndex`] with uint8 scalar quantization from `innr::scalar`.
//! Graph construction uses full-precision f32 vectors. Search walks the
//! HNSW graph using asymmetric distance (f32 query vs u8 stored vectors),
//! giving ~4x memory reduction on vector storage with typically <2% recall loss.
//!
//! # Two-stage search
//!
//! 1. Graph traversal with asymmetric u8 distance (cheap, approximate)
//! 2. Optional reranking of top candidates with exact f32 distance
//!
//! # Example
//!
//! ```rust,no_run
//! # fn main() -> Result<(), vicinity::RetrieveError> {
//! use vicinity::hnsw::scalar_quantized::ScalarQuantizedHNSW;
//!
//! let dim = 128;
//! let mut sq = ScalarQuantizedHNSW::new(dim, 16, 16)?;
//!
//! // Add normalized vectors (same as HNSWIndex)
//! let v = vicinity::distance::normalize(&vec![0.1; dim]);
//! sq.add_slice(0, &v)?;
//! // ... add more vectors ...
//!
//! // Build graph (f32) then quantize
//! sq.build()?;
//!
//! // Search with asymmetric quantized distance
//! let q = vicinity::distance::normalize(&vec![0.15; dim]);
//! let results = sq.search(&q, 10, 50)?;
//!
//! // Or search + rerank top candidates with exact f32
//! let results = sq.search_reranked(&q, 10, 50, 100)?;
//! # Ok(())
//! # }
//! ```

use crate::hnsw::graph::{HNSWIndex, Layer};
use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet};

/// HNSW index with scalar quantization (SQ8).
///
/// Stores quantized u8 vectors alongside the HNSW graph. Graph construction
/// uses f32 vectors (for quality). Search uses asymmetric distance: the query
/// stays in f32 while graph vectors are u8.
///
/// Quantization uses per-dimension min/max ranges (FAISS `QT_8bit` style),
/// which gives better accuracy than a single global range when dimensions
/// have different scales.
///
/// Memory for vector storage: `n * dim` bytes (u8) vs `n * dim * 4` bytes (f32).
/// The HNSW graph edges are unchanged.
pub struct ScalarQuantizedHNSW {
    /// The underlying HNSW index (owns graph + f32 vectors).
    index: HNSWIndex,
    /// Flat quantized storage: `quantized[i * dim .. (i+1) * dim]` for internal id `i`.
    quantized: Vec<u8>,
    /// Per-dimension quantization scale (alpha = max - min per dim). Length = dimension.
    scales: Vec<f32>,
    /// Per-dimension quantization offset (min per dim). Length = dimension.
    offsets: Vec<f32>,
    /// Whether quantization has been performed.
    quantized_built: bool,
}

impl ScalarQuantizedHNSW {
    /// Create a new scalar-quantized HNSW index.
    ///
    /// Parameters are the same as [`HNSWIndex::new`]: `dimension`, `m` (edges per node),
    /// `m_max` (max edges per node).
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        let index = HNSWIndex::new(dimension, m, m_max)?;
        Ok(Self {
            index,
            quantized: Vec::new(),
            scales: Vec::new(),
            offsets: Vec::new(),
            quantized_built: false,
        })
    }

    /// Add a vector. Must be L2-normalized (same requirement as [`HNSWIndex`]).
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        self.index.add_slice(doc_id, vector)
    }

    /// Build the HNSW graph (f32) and then quantize all vectors to u8.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        self.index.build()?;
        self.quantize_vectors();
        Ok(())
    }

    /// Quantize all vectors from the built index using per-dimension min/max ranges.
    fn quantize_vectors(&mut self) {
        let n = self.index.num_vectors;
        let dim = self.index.dimension;

        // Compute per-dimension min/max.
        let mut mins = vec![f32::MAX; dim];
        let mut maxs = vec![f32::MIN; dim];
        for i in 0..n {
            let base = i * dim;
            for d in 0..dim {
                let v = self.index.vectors[base + d];
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        // scales[d] = max[d] - min[d]; offsets[d] = min[d].
        // Guard against zero range (constant dimension).
        self.scales = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                let alpha = mx - mn;
                if alpha > 0.0 {
                    alpha
                } else {
                    1.0
                }
            })
            .collect();
        self.offsets = mins;

        // Quantize each vector element with its per-dimension params.
        self.quantized = Vec::with_capacity(n * dim);
        for i in 0..n {
            let base = i * dim;
            for d in 0..dim {
                let v = self.index.vectors[base + d];
                let normalized = (v - self.offsets[d]) * (255.0 / self.scales[d]);
                self.quantized
                    .push(normalized.round().clamp(0.0, 255.0) as u8);
            }
        }

        self.quantized_built = true;
    }

    /// Search using asymmetric quantized distance (f32 query vs u8 vectors).
    ///
    /// Walks the HNSW graph using `1 - asymmetric_dot(q, dequant(v))` as the
    /// distance function. Returns `(doc_id, distance)` pairs sorted by distance.
    ///
    /// This is faster and uses less memory bandwidth than f32 search, at the cost
    /// of slightly lower recall.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.check_search_ready(query)?;

        let results = self.search_quantized_graph(query, ef)?;

        // Take top-k, map internal ids to doc ids
        let mut output: Vec<(u32, f32)> = results
            .into_iter()
            .take(k)
            .map(|(internal_id, dist)| (self.index.doc_ids[internal_id as usize], dist))
            .collect();
        output.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output)
    }

    /// Search with oversampling + exact f32 reranking.
    ///
    /// 1. Retrieve `rerank_pool` candidates using quantized graph search
    /// 2. Rerank using exact f32 cosine distance
    /// 3. Return top `k`
    ///
    /// Higher `rerank_pool` improves recall at the cost of more f32 distance
    /// computations. A typical ratio is `rerank_pool = 3 * k` to `10 * k`.
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

        // Rerank with exact f32 cosine distance
        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(pool)
            .map(|(internal_id, _approx_dist)| {
                let vec = self.index.get_vector(internal_id as usize);
                let exact_dist = crate::distance::cosine_distance_normalized(query, vec);
                (self.index.doc_ids[internal_id as usize], exact_dist)
            })
            .collect();

        reranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        reranked.truncate(k);
        Ok(reranked)
    }

    /// Memory statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        let dim = self.index.dimension;
        let n = self.index.num_vectors;
        let f32_bytes = self.index.vectors.len() * 4;
        let u8_bytes = self.quantized.len();
        // Params overhead: 2 * dim f32s (scales + offsets).
        let params_bytes = dim * 4 * 2;

        MemoryStats {
            num_vectors: n,
            dimension: dim,
            f32_vector_bytes: f32_bytes,
            u8_vector_bytes: u8_bytes + params_bytes,
            compression_ratio: if u8_bytes > 0 {
                f32_bytes as f64 / (u8_bytes + params_bytes) as f64
            } else {
                0.0
            },
        }
    }

    /// Access the underlying HNSW index (e.g., for serialization or direct f32 search).
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

    /// Precompute per-query context for asymmetric distance.
    ///
    /// `q_scaled[i] = q[i] * scales[i] / 255.0` so that
    /// `dot(q, dequant(d)) = sum(q_scaled[i] * d_u8[i]) + offset_dot`.
    fn precompute_query_context(&self, query: &[f32]) -> QueryCtx {
        let dim = self.index.dimension;
        let mut q_scaled = Vec::with_capacity(dim);
        let mut offset_dot: f32 = 0.0;
        for (d, &q) in query.iter().enumerate().take(dim) {
            q_scaled.push(q * self.scales[d] / 255.0);
            offset_dot += q * self.offsets[d];
        }
        QueryCtx {
            q_scaled,
            offset_dot,
        }
    }

    /// Walk the HNSW graph using asymmetric quantized cosine distance.
    ///
    /// Returns `(internal_id, distance)` sorted by distance.
    fn search_quantized_graph(
        &self,
        query: &[f32],
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let ctx = self.precompute_query_context(query);

        // Find entry point (highest-layer node).
        let (entry_point, entry_layer) = self.find_entry_point();

        // Navigate upper layers (greedy single-node descent).
        let mut current = entry_point;
        let mut current_dist = self.quantized_cosine_distance(&ctx, current);

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
                    let dist = self.quantized_cosine_distance(&ctx, neighbor_id);
                    if dist < current_dist {
                        current_dist = dist;
                        current = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Beam search in base layer (layer 0) with quantized distance.
        if self.index.layers.is_empty() {
            return Ok(Vec::new());
        }
        let base_layer = &self.index.layers[0];
        let results = self.beam_search_quantized(&ctx, current, base_layer, ef);
        Ok(results)
    }

    /// Asymmetric cosine distance: `1 - dot(query_f32, dequant(vec_u8))`.
    ///
    /// Per-dimension decomposition:
    /// `dot = sum(q_scaled[i] * d_u8[i]) + offset_dot`
    /// where `q_scaled` and `offset_dot` are precomputed per query.
    #[inline]
    fn quantized_cosine_distance(&self, ctx: &QueryCtx, internal_id: u32) -> f32 {
        let dim = self.index.dimension;
        let start = internal_id as usize * dim;
        let end = start + dim;
        let quantized_vec = &self.quantized[start..end];

        let mixed: f32 = ctx
            .q_scaled
            .iter()
            .zip(quantized_vec.iter())
            .map(|(&q, &d)| q * d as f32)
            .sum();

        1.0 - (mixed + ctx.offset_dot)
    }

    /// Beam search in a single layer using quantized distance.
    fn beam_search_quantized(
        &self,
        ctx: &QueryCtx,
        entry_point: u32,
        layer: &Layer,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        let n = self.index.num_vectors;

        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        // Visited set: dense for small indexes, sparse for large.
        let mut visited = if n <= 100_000 {
            Visited::Dense(vec![false; n])
        } else {
            Visited::Sparse(HashSet::with_capacity(ef * 2))
        };

        let entry_dist = self.quantized_cosine_distance(ctx, entry_point);
        candidates.push(MinCandidate {
            id: entry_point,
            distance: entry_dist,
        });
        results.push(MaxResult {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);

        while let Some(candidate) = candidates.pop() {
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if candidate.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = layer.get_neighbors(candidate.id);
            for &neighbor_id in neighbors.iter() {
                if visited.insert(neighbor_id) {
                    let dist = self.quantized_cosine_distance(ctx, neighbor_id);

                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                    if results.len() < ef || dist < worst_dist {
                        candidates.push(MinCandidate {
                            id: neighbor_id,
                            distance: dist,
                        });
                        results.push(MaxResult {
                            id: neighbor_id,
                            distance: dist,
                        });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_by(|a, b| a.1.total_cmp(&b.1));
        output
    }

    fn find_entry_point(&self) -> (u32, usize) {
        let mut ep = 0u32;
        let mut el = 0u8;
        for (idx, &layer) in self.index.layer_assignments.iter().enumerate() {
            if layer > el {
                ep = idx as u32;
                el = layer;
            }
        }
        (ep, el as usize)
    }
}

/// Memory usage statistics.
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Number of indexed vectors.
    pub num_vectors: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Memory used by f32 vector storage (bytes).
    pub f32_vector_bytes: usize,
    /// Memory used by u8 quantized storage (bytes, including params).
    pub u8_vector_bytes: usize,
    /// Compression ratio (f32 / u8).
    pub compression_ratio: f64,
}

/// Precomputed per-query context for asymmetric distance with per-dimension quantization.
///
/// `q_scaled[i] = query[i] * scales[i] / 255.0` absorbs the per-dimension scale
/// into the query so that the inner loop is a plain mixed dot product.
/// `offset_dot = sum(query[i] * offsets[i])` is the constant bias term.
struct QueryCtx {
    q_scaled: Vec<f32>,
    offset_dot: f32,
}

// ── Heap helpers (local to this module) ─────────────────────────────────────

#[derive(PartialEq)]
struct MinCandidate {
    id: u32,
    distance: f32,
}
impl Eq for MinCandidate {}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.total_cmp(&self.distance)
    }
}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(PartialEq)]
struct MaxResult {
    id: u32,
    distance: f32,
}
impl Eq for MaxResult {}
impl Ord for MaxResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}
impl PartialOrd for MaxResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Visited-node tracker.
enum Visited {
    Dense(Vec<bool>),
    Sparse(HashSet<u32>),
}

impl Visited {
    /// Mark as visited. Returns `true` if newly inserted.
    fn insert(&mut self, id: u32) -> bool {
        match self {
            Visited::Dense(v) => {
                let idx = id as usize;
                if idx < v.len() && !v[idx] {
                    v[idx] = true;
                    true
                } else {
                    false
                }
            }
            Visited::Sparse(s) => s.insert(id),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::distance::normalize;

    fn make_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // Simple deterministic pseudo-random (not cryptographic).
        let mut state = seed;
        (0..n)
            .map(|_| {
                let raw: Vec<f32> = (0..dim)
                    .map(|_| {
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (state >> 33) as f32 / (1u64 << 31) as f32 - 1.0
                    })
                    .collect();
                normalize(&raw)
            })
            .collect()
    }

    #[test]
    fn test_basic_search() {
        let dim = 32;
        let n = 200;
        let vectors = make_random_vectors(n, dim, 42);

        let mut sq = ScalarQuantizedHNSW::new(dim, 16, 16).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            sq.add_slice(i as u32, v).unwrap();
        }
        sq.build().unwrap();

        // Search should return results
        let results = sq.search(&vectors[0], 10, 50).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // First result should be the query itself (or very close)
        assert_eq!(results[0].0, 0, "nearest neighbor of v[0] should be v[0]");
    }

    #[test]
    fn test_search_reranked() {
        let dim = 32;
        let n = 200;
        let vectors = make_random_vectors(n, dim, 123);

        let mut sq = ScalarQuantizedHNSW::new(dim, 16, 16).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            sq.add_slice(i as u32, v).unwrap();
        }
        sq.build().unwrap();

        let results = sq.search_reranked(&vectors[5], 10, 50, 50).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Reranked first result should be the query vector itself
        assert_eq!(results[0].0, 5);
    }

    #[test]
    fn test_memory_stats() {
        let dim = 64;
        let n = 100;
        let vectors = make_random_vectors(n, dim, 99);

        let mut sq = ScalarQuantizedHNSW::new(dim, 8, 8).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            sq.add_slice(i as u32, v).unwrap();
        }
        sq.build().unwrap();

        let stats = sq.memory_stats();
        assert_eq!(stats.num_vectors, n);
        assert_eq!(stats.dimension, dim);
        assert_eq!(stats.f32_vector_bytes, n * dim * 4);
        // u8 storage: n * dim + dim * 8 bytes for per-dim params (scales + offsets)
        assert_eq!(stats.u8_vector_bytes, n * dim + dim * 8);
        // Compression ratio: n*dim*4 / (n*dim + dim*8).
        // For n=100, dim=64: 25600 / (6400 + 512) = 3.70
        let expected = (n * dim * 4) as f64 / (n * dim + dim * 8) as f64;
        assert!(
            (stats.compression_ratio - expected).abs() < 0.01,
            "compression ratio {} should be ~{:.2}",
            stats.compression_ratio,
            expected
        );
    }

    #[test]
    fn test_recall_vs_exact() {
        // Compare SQ8 search recall against exact HNSW search.
        let dim = 32;
        let n = 500;
        let vectors = make_random_vectors(n, dim, 777);

        let mut sq = ScalarQuantizedHNSW::new(dim, 16, 16).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            sq.add_slice(i as u32, v).unwrap();
        }
        sq.build().unwrap();

        let k = 10;
        let ef = 100;
        let num_queries = 20;

        let mut total_recall = 0.0;
        for qi in 0..num_queries {
            let query = &vectors[qi * 5];

            // Exact HNSW search
            let exact = sq.inner().search(query, k, ef).unwrap();
            let exact_ids: std::collections::HashSet<u32> =
                exact.iter().map(|&(id, _)| id).collect();

            // SQ8 reranked search
            let sq_results = sq.search_reranked(query, k, ef, k * 5).unwrap();
            let sq_ids: std::collections::HashSet<u32> =
                sq_results.iter().map(|&(id, _)| id).collect();

            let overlap = exact_ids.intersection(&sq_ids).count();
            total_recall += overlap as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(
            avg_recall > 0.8,
            "SQ8 reranked recall {:.3} should be > 0.8",
            avg_recall
        );
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut sq = ScalarQuantizedHNSW::new(32, 8, 8).unwrap();
        let v = normalize(&vec![0.1; 32]);
        sq.add_slice(0, &v).unwrap();
        sq.build().unwrap();

        let bad_query = vec![0.1; 64];
        let err = sq.search(&bad_query, 10, 50);
        assert!(err.is_err());
    }

    #[test]
    fn test_search_before_build_fails() {
        let sq = ScalarQuantizedHNSW::new(32, 8, 8).unwrap();
        let q = normalize(&vec![0.1; 32]);
        let err = sq.search(&q, 10, 50);
        assert!(err.is_err());
    }
}
