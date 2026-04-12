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

use crate::hnsw::graph::{HNSWIndex, Layer};
use crate::RetrieveError;
use qntz::rabitq::{QuantizedVector, RaBitQConfig, RaBitQQuantizer};
use std::collections::{BinaryHeap, HashSet};

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
    /// Rotation matrix (reproduced from seed, same as quantizer's).
    rotation: Vec<f32>,
    /// Centroid (computed from training data).
    centroid: Vec<f32>,
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
            rotation: Vec::new(),
            centroid: Vec::new(),
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

        // Store rotation matrix (reproduced from same seed) and centroid
        // so we can pre-rotate queries without calling back into the quantizer.
        self.rotation = generate_orthogonal_rotation(dim, self.seed);
        self.centroid = compute_centroid(&self.index.vectors, n, dim);
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

        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(pool)
            .map(|(internal_id, _approx_dist)| {
                let vec = self.index.get_vector(internal_id as usize);
                let exact_dist = crate::distance::cosine_distance_normalized(query, vec);
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
    fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        let dim = self.index.dimension;
        let residual: Vec<f32> = query
            .iter()
            .zip(self.centroid.iter())
            .map(|(q, c)| q - c)
            .collect();
        apply_rotation(&residual, &self.rotation, dim)
    }

    /// Walk the HNSW graph using RaBitQ approximate distance.
    fn search_quantized_graph(
        &self,
        query: &[f32],
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let rotated_query = self.rotate_query(query);

        // Find entry point (highest-layer node).
        let (entry_point, entry_layer) = self.find_entry_point();

        // Navigate upper layers with greedy single-node descent.
        let mut current = entry_point;
        let mut current_dist = rabitq_distance(&rotated_query, &self.codes[current as usize]);

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
                    let dist = rabitq_distance(&rotated_query, &self.codes[neighbor_id as usize]);
                    if dist < current_dist {
                        current_dist = dist;
                        current = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Beam search in base layer with quantized distance.
        if self.index.layers.is_empty() {
            return Ok(Vec::new());
        }
        let base_layer = &self.index.layers[0];
        Ok(self.beam_search_quantized(&rotated_query, current, base_layer, ef))
    }

    /// Beam search in a single layer using quantized distance.
    fn beam_search_quantized(
        &self,
        rotated_query: &[f32],
        entry_point: u32,
        layer: &Layer,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        let n = self.index.num_vectors;

        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        let mut visited = if n <= 100_000 {
            Visited::Dense(vec![false; n])
        } else {
            Visited::Sparse(HashSet::with_capacity(ef * 2))
        };

        let entry_dist = rabitq_distance(rotated_query, &self.codes[entry_point as usize]);
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
                    let dist = rabitq_distance(rotated_query, &self.codes[neighbor_id as usize]);

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
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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

// ── RaBitQ distance (inline, no quantizer reference needed) ──────────────────

/// Approximate L2 distance from a pre-rotated query to a quantized vector.
#[inline]
fn rabitq_distance(rotated_query: &[f32], qv: &QuantizedVector) -> f32 {
    let cb = -((1 << qv.ex_bits) as f32 - 0.5);
    let mut ip = 0.0f32;
    for (&q, &c) in rotated_query.iter().zip(qv.codes.iter()) {
        ip += q * (c as f32 + cb);
    }
    (qv.f_add + qv.f_rescale * ip).max(0.0).sqrt()
}

// ── Rotation matrix (deterministic from seed, matches qntz) ─────────────────

fn generate_orthogonal_rotation(dimension: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut rotation = vec![0.0f32; dimension * dimension];
    let mut state = seed;
    let mut next_rand = || -> f32 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
    };

    let mut basis: Vec<Vec<f32>> = Vec::new();
    for i in 0..dimension {
        let mut v: Vec<f32> = (0..dimension).map(|_| next_rand()).collect();
        for b in &basis {
            let dot: f32 = v.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= dot * bi;
            }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for vi in &mut v {
                *vi /= norm;
            }
            basis.push(v);
        } else {
            let mut v = vec![0.0f32; dimension];
            v[i] = 1.0;
            basis.push(v);
        }
    }
    for (i, row) in basis.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            rotation[i * dimension + j] = val;
        }
    }
    rotation
}

fn apply_rotation(vector: &[f32], rotation: &[f32], dimension: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dimension];
    for (i, out) in result.iter_mut().enumerate() {
        let row_start = i * dimension;
        let mut sum = 0.0f32;
        for j in 0..dimension {
            sum += rotation[row_start + j] * vector[j];
        }
        *out = sum;
    }
    result
}

fn compute_centroid(vectors: &[f32], n: usize, dim: usize) -> Vec<f32> {
    let mut centroid = vec![0.0f32; dim];
    for i in 0..n {
        let base = i * dim;
        for d in 0..dim {
            centroid[d] += vectors[base + d];
        }
    }
    let inv_n = 1.0 / n as f32;
    for c in &mut centroid {
        *c *= inv_n;
    }
    centroid
}

// ── Helper types ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct MinCandidate {
    id: u32,
    distance: f32,
}
impl Eq for MinCandidate {}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.total_cmp(&self.distance)
    }
}

#[derive(Clone, Copy, PartialEq)]
struct MaxResult {
    id: u32,
    distance: f32,
}
impl Eq for MaxResult {}
impl PartialOrd for MaxResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

enum Visited {
    Dense(Vec<bool>),
    Sparse(HashSet<u32>),
}

impl Visited {
    fn insert(&mut self, id: u32) -> bool {
        match self {
            Visited::Dense(v) => {
                let idx = id as usize;
                if v[idx] {
                    false
                } else {
                    v[idx] = true;
                    true
                }
            }
            Visited::Sparse(s) => s.insert(id),
        }
    }
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
        // Verify our inline rabitq_distance matches qntz's approximate_l2_sqr
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

        // Our rotation matrix (same seed)
        let rotation = super::generate_orthogonal_rotation(dim, seed);
        let centroid = super::compute_centroid(&flat, n, dim);

        let query = &vectors[0];

        // qntz distance
        let qntz_dist = quantizer.approximate_l2_sqr(query, &codes[1]).unwrap();

        // Our inline distance
        let residual: Vec<f32> = query
            .iter()
            .zip(centroid.iter())
            .map(|(q, c)| q - c)
            .collect();
        let rotated = super::apply_rotation(&residual, &rotation, dim);
        let our_dist_sqr = {
            let qv = &codes[1];
            let cb = -((1 << qv.ex_bits) as f32 - 0.5);
            let mut ip = 0.0f32;
            for (&q, &c) in rotated.iter().zip(qv.codes.iter()) {
                ip += q * (c as f32 + cb);
            }
            (qv.f_add + qv.f_rescale * ip).max(0.0)
        };

        let diff = (qntz_dist - our_dist_sqr).abs();
        assert!(
            diff < 1e-4,
            "distance mismatch: qntz={qntz_dist}, ours={our_dist_sqr}, diff={diff}"
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
}
