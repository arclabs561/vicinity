//! SQ4U: HNSW with 4-bit scalar quantized graph traversal.
//!
//! During beam search, approximate distances are computed via a precomputed
//! lookup table over 4-bit codes, replacing the O(d) f32 multiply-accumulate
//! with O(d/2) table lookups + additions. Final top-k results are reranked
//! with exact f32 distance.
//!
//! Compared to SymphonyQG (RaBitQ), SQ4U avoids the O(d^2) query rotation --
//! the per-query table precomputation is O(d * 16) and the per-neighbor
//! distance is pure table lookup over packed nibbles.
//!
//! # Two-stage search
//!
//! 1. Graph traversal with SQ4 approximate L2 distance (table lookup)
//! 2. Reranking of top candidates with exact f32 distance
//!
//! # Example
//!
//! ```rust,ignore
//! # fn main() -> Result<(), vicinity::RetrieveError> {
//! use vicinity::hnsw::sq4u::HNSWSq4Index;
//!
//! let dim = 128;
//! let mut index = HNSWSq4Index::new(dim, 16, 16)?;
//!
//! let v = vicinity::distance::normalize(&vec![0.1; dim]);
//! index.add_slice(0, &v)?;
//! // ... add more vectors ...
//!
//! index.build()?;
//!
//! // Search with quantized traversal + exact reranking
//! let q = vicinity::distance::normalize(&vec![0.15; dim]);
//! let results = index.search_reranked(&q, 10, 50, 100)?;
//! # Ok(())
//! # }
//! ```

use crate::hnsw::graph::HNSWIndex;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// HNSW index with 4-bit scalar quantized graph traversal.
///
/// Graph construction uses full-precision f32 vectors. Search walks the graph
/// using a precomputed distance table over 4-bit codes, then reranks the top
/// candidates with exact distance.
///
/// Memory: f32 vectors (for reranking) + 0.5 bytes/dim quantized codes.
///
/// # Status: Experimental
///
/// SQ4U is 2-3x slower than plain HNSW on tested datasets due to the
/// mandatory rerank pass. However, it is the only quantized traversal
/// option that works natively with L2/unnormalized data. SymphonyQG
/// (RaBitQ) currently requires cosine/normalized vectors because it uses
/// a global centroid; vertex-relative normalization (per the SymphonyQG
/// paper, arXiv:2411.12229) is needed to support L2 at scale.
pub struct HNSWSq4Index {
    /// The underlying HNSW index (owns graph + f32 vectors).
    index: HNSWIndex,
    /// Packed 4-bit codes, one entry per vector. Each entry is ceil(d/2) bytes.
    codes: Vec<Vec<u8>>,
    /// Per-dimension minimum (length d).
    mins: Vec<f32>,
    /// Per-dimension step: (max - min) / 15 (length d).
    steps: Vec<f32>,
    /// Per-dimension inverse scale: 15 / (max - min) (length d).
    inv_scales: Vec<f32>,
    /// Whether quantization has been performed.
    built: bool,
}

impl HNSWSq4Index {
    /// Create a new SQ4U index with default cosine metric.
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

    /// Create a new SQ4U index with explicit HNSW parameters.
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

    /// Build the HNSW graph and quantize all vectors to 4-bit codes.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        self.index.build()?;
        self.quantize_vectors()?;
        self.built = true;
        Ok(())
    }

    /// Save a built SQ4U index to a directory snapshot.
    ///
    /// The HNSW graph and full-precision vectors are the source of truth.
    /// Quantized codes are rebuilt on load from the post-build HNSW vector
    /// order so the auxiliary arrays cannot drift from graph node IDs.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter("index not built".into()));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        crate::graph_snapshot::write_json_atomic(
            &output_dir.join("manifest.json"),
            &Sq4uSnapshotManifest {
                magic: *b"VICSQ4U1",
                version: 1,
            },
        )?;
        self.index.save_to_file(output_dir.join("hnsw.json"))?;
        Ok(())
    }

    /// Load an SQ4U index from a directory snapshot.
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let manifest: Sq4uSnapshotManifest =
            crate::graph_snapshot::read_json(&input_dir.join("manifest.json"))?;
        if manifest.magic != *b"VICSQ4U1" {
            return Err(RetrieveError::FormatError(
                "invalid SQ4U snapshot magic".into(),
            ));
        }
        if manifest.version != 1 {
            return Err(RetrieveError::FormatError(format!(
                "unsupported SQ4U snapshot version {}",
                manifest.version
            )));
        }

        let index = HNSWIndex::load_from_file(input_dir.join("hnsw.json"))?;
        let mut loaded = Self {
            index,
            codes: Vec::new(),
            mins: Vec::new(),
            steps: Vec::new(),
            inv_scales: Vec::new(),
            built: false,
        };
        loaded.quantize_vectors()?;
        loaded.built = true;
        Ok(loaded)
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
    /// 2. Compute exact f32 L2 distance for each
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
    /// then pack each vector into 4-bit codes.
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
                steps[d] = range / 15.0;
                inv_scales[d] = 15.0 / range;
            }
        }

        // Pack each vector.
        let code_len = dim.div_ceil(2);
        let mut codes = Vec::with_capacity(n);
        let mut buf = vec![0u8; code_len];
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            crate::sq4::pack_vector(v, &mins, &inv_scales, &mut buf);
            codes.push(buf.clone());
        }

        self.codes = codes;
        self.mins = mins;
        self.steps = steps;
        self.inv_scales = inv_scales;
        Ok(())
    }

    /// Precompute the distance table for a query: table[d][code] = (q[d] - decoded)^2.
    /// Flattened as [d * 16] f32 values.
    #[inline]
    #[allow(clippy::needless_range_loop)]
    fn precompute_table(&self, query: &[f32]) -> Vec<f32> {
        let dim = self.index.dimension;
        let mut table = vec![0.0f32; dim * 16];
        for d in 0..dim {
            let q = query[d];
            let min = self.mins[d];
            let step = self.steps[d];
            let base = d * 16;
            for code in 0..16u32 {
                let decoded = min + code as f32 * step;
                let diff = q - decoded;
                table[base + code as usize] = diff * diff;
            }
        }
        table
    }

    /// Approximate L2^2 distance using the precomputed table.
    #[inline]
    fn approx_dist_table(table: &[f32], code: &[u8], dim: usize) -> f32 {
        let mut sum = 0.0f32;
        let pairs = dim / 2;
        for p in 0..pairs {
            let byte = code[p];
            let lo = (byte & 0x0F) as usize;
            let hi = (byte >> 4) as usize;
            sum += table[2 * p * 16 + lo] + table[(2 * p + 1) * 16 + hi];
        }
        if dim % 2 == 1 {
            let lo = (code[pairs] & 0x0F) as usize;
            sum += table[(dim - 1) * 16 + lo];
        }
        sum
    }

    /// Walk the HNSW graph using SQ4 approximate distance.
    fn search_quantized(&self, query: &[f32], ef: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let table = self.precompute_table(query);
        let codes = &self.codes;
        let dim = self.index.dimension;

        let (entry_point, entry_layer) = self.index.entry_point().unwrap_or((0, 0));

        // Navigate upper layers with greedy single-node descent.
        let mut current = entry_point;
        let mut current_dist = Self::approx_dist_table(&table, &codes[current as usize], dim);

        for layer_idx in (1..=entry_layer).rev() {
            if layer_idx >= self.index.layers.len() {
                continue;
            }
            let layer = &self.index.layers[layer_idx];
            let mut changed = true;
            while changed {
                changed = false;
                for &neighbor_id in layer.get_neighbors(current).iter() {
                    let dist = Self::approx_dist_table(&table, &codes[neighbor_id as usize], dim);
                    if dist < current_dist {
                        current_dist = dist;
                        current = neighbor_id;
                        changed = true;
                    }
                }
            }
        }

        // Base layer: beam search with table-lookup distance.
        if self.index.layers.is_empty() {
            return Ok(Vec::new());
        }
        let base_layer = &self.index.layers[0];
        let dist_fn = |_q: &[f32], node_id: u32| -> f32 {
            Self::approx_dist_table(&table, &codes[node_id as usize], dim)
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Sq4uSnapshotManifest {
    magic: [u8; 8],
    version: u32,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn random_normalized(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() - 0.5).collect();
                crate::distance::normalize(&v)
            })
            .collect()
    }

    #[test]
    fn sq4u_search_reranked_recall() {
        let dim = 32;
        let n = 300;
        let k = 10;
        let vecs = random_normalized(n, dim, 42);

        let mut index = HNSWSq4Index::new(dim, 16, 32).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        // Brute-force ground truth
        let query = &vecs[0];
        let mut gt: Vec<(u32, f32)> = vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i as u32,
                    crate::distance::cosine_distance_normalized(query, v),
                )
            })
            .collect();
        gt.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt_ids: std::collections::HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();

        let results = index.search_reranked(query, k, 64, 50).unwrap();
        let result_ids: std::collections::HashSet<u32> =
            results.iter().map(|(id, _)| *id).collect();

        let recall = gt_ids.intersection(&result_ids).count() as f32 / k as f32;
        assert!(
            recall >= 0.50,
            "SQ4U reranked recall@{k} = {recall:.3}, expected >= 0.50"
        );
    }

    #[test]
    fn sq4u_approx_closer_than_random() {
        let dim = 64;
        let n = 100;
        let vecs = random_normalized(n, dim, 99);

        let mut index = HNSWSq4Index::new(dim, 16, 32).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query = &vecs[0];
        let results = index.search(query, 5, 64).unwrap();
        // The query itself should be among the top results (self-retrieval)
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "Query vector should be in its own top-5 SQ4U results"
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_reranked_search() {
        let dim = 32;
        let vecs = random_normalized(120, dim, 123);

        let mut index = HNSWSq4Index::new(dim, 16, 32).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add_slice(i as u32, v).unwrap();
        }
        index.build().unwrap();

        let before = index.search_reranked(&vecs[0], 8, 64, 50).unwrap();
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = HNSWSq4Index::load_from_dir(dir.path()).unwrap();
        let after = loaded.search_reranked(&vecs[0], 8, 64, 50).unwrap();

        assert_eq!(after, before);
    }
}
