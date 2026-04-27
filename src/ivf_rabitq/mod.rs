//! IVF-RaBitQ: Inverted File with Randomized Binary Quantization.
//!
//! Combines IVF partitioning with RaBitQ quantization for memory-efficient ANN
//! search. Unlike IVF-PQ, RaBitQ requires no codebook training, only a random
//! rotation matrix and per-vector correction factors.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.7", features = ["ivf_rabitq"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::ivf_rabitq::{IVFRaBitQIndex, IVFRaBitQParams};
//!
//! let params = IVFRaBitQParams {
//!     num_clusters: 256,
//!     nprobe: 10,
//!     ..Default::default()
//! };
//!
//! let mut index = IVFRaBitQIndex::new(128, params)?;
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Memory Comparison (d=128, n=1M)
//!
//! | Method | Bytes/vector | Total | Recall@10 |
//! |--------|-------------|-------|-----------|
//! | Raw f32 | 512 | 512 MB | 100% |
//! | IVF-PQ (M=8) | 8 | 8 MB | ~85% |
//! | IVF-RaBitQ (4-bit) | 68 | 68 MB | ~95% |
//! | IVF-RaBitQ (1-bit) | 20 | 20 MB | ~80% |
//!
//! The 4-bit variant uses more memory than PQ but achieves higher recall because
//! RaBitQ preserves per-dimension information (vs PQ's subspace compression).
//! The 1-bit variant is competitive with PQ on memory while offering faster
//! distance computation via popcount.
//!
//! # How It Works
//!
//! 1. **IVF partitioning**: k-means clusters vectors into cells
//! 2. **Residual computation**: subtract cluster centroid from each vector
//! 3. **RaBitQ quantization**: random rotation + sign/extended bits + correction factors
//! 4. **Search**: probe nearest centroids, scan clusters with RaBitQ approximate distances
//!
//! Distance is computed asymmetrically: query is exact f32, database vectors are
//! quantized. The correction factors (`f_add`, `f_rescale`) provide theoretical
//! error bounds on the distance estimate.
//!
//! # References
//!
//! - Gao et al. (2024). "RaBitQ: Quantizing High-Dimensional Vectors with a
//!   Theoretical Error Bound for Approximate Nearest Neighbor Search." SIGMOD 2024.
//! - Chen et al. (2026). "IVF-RaBitQ: GPU-native IVF with RaBitQ." arXiv:2602.23999.

use crate::distance::FloatOrd;
use crate::RetrieveError;
use qntz::rabitq::{RaBitQConfig, RaBitQQuantizer};

/// IVF-RaBitQ parameters.
#[derive(Clone, Debug)]
pub struct IVFRaBitQParams {
    /// Number of IVF clusters (inverted lists).
    pub num_clusters: usize,
    /// Number of clusters to probe during search.
    pub nprobe: usize,
    /// RaBitQ bits per dimension (1-8). Default: 4.
    pub total_bits: usize,
    /// Random seed for the rotation matrix.
    pub seed: u64,
}

impl Default for IVFRaBitQParams {
    fn default() -> Self {
        Self {
            num_clusters: 256,
            nprobe: 10,
            total_bits: 4,
            seed: 42,
        }
    }
}

/// A cluster (inverted list) storing RaBitQ-quantized residual vectors.
///
/// Vectors are quantized via qntz's edge API: each `v` is stored as an
/// [`EdgeQuantizedVector`](qntz::rabitq::EdgeQuantizedVector) whose "parent"
/// is the cluster centroid. At search time the caller adds the per-cluster
/// `||q - cluster_c||^2` to each edge term, yielding an absolute distance
/// that is consistent across clusters (and therefore rankable against other
/// probed clusters in a shared heap).
#[derive(Debug)]
struct Cluster {
    /// Indices into the global vector array (insertion order).
    vector_indices: Vec<u32>,
    /// Edge-quantized residuals (`v - cluster_c`) for each vector in this cluster.
    quantized: Vec<qntz::rabitq::EdgeQuantizedVector>,
}

/// IVF-RaBitQ index.
pub struct IVFRaBitQIndex {
    dimension: usize,
    params: IVFRaBitQParams,
    built: bool,
    compacted: bool,

    // Raw vectors (kept for build and exact reranking; drop with compact() to save memory)
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    // IVF components
    clusters: Vec<Cluster>,
    /// Flat centroid storage: `[c0_d0, c0_d1, ..., c1_d0, ...]`.
    centroids: Vec<f32>,

    // RaBitQ quantizer (shared rotation matrix across all clusters)
    quantizer: RaBitQQuantizer,

    /// HNSW coarse quantizer over centroids for O(log k) lookup.
    #[cfg(feature = "hnsw")]
    coarse_quantizer: Option<crate::hnsw::HNSWIndex>,
}

impl IVFRaBitQIndex {
    /// Create a new IVF-RaBitQ index.
    pub fn new(dimension: usize, params: IVFRaBitQParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        let config = RaBitQConfig {
            total_bits: params.total_bits,
            t_const: None,
        };
        let quantizer = RaBitQQuantizer::with_config(dimension, params.seed, config)
            .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ config: {e}")))?;

        Ok(Self {
            dimension,
            params,
            built: false,
            compacted: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            clusters: Vec::new(),
            centroids: Vec::new(),
            quantizer,
            #[cfg(feature = "hnsw")]
            coarse_quantizer: None,
        })
    }

    /// Set the number of clusters to probe during search.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.params.nprobe = nprobe;
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector from a borrowed slice.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after index is built".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        // L2-normalize on insertion (cosine index)
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

    /// Add a batch of vectors.
    pub fn add_batch(&mut self, doc_ids: &[u32], vectors: &[f32]) -> Result<(), RetrieveError> {
        if vectors.len() != doc_ids.len() * self.dimension {
            return Err(RetrieveError::InvalidParameter(format!(
                "expected {} floats for {} vectors of dim {}, got {}",
                doc_ids.len() * self.dimension,
                doc_ids.len(),
                self.dimension,
                vectors.len()
            )));
        }
        for (i, &doc_id) in doc_ids.iter().enumerate() {
            let start = i * self.dimension;
            let end = start + self.dimension;
            self.add_slice(doc_id, &vectors[start..end])?;
        }
        Ok(())
    }

    /// Build the index: cluster vectors, quantize residuals.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Stage 1: k-means clustering
        let num_clusters = self.params.num_clusters.min(self.num_vectors);
        let mut kmeans = crate::partitioning::kmeans::KMeans::new(self.dimension, num_clusters)?;
        kmeans.fit(&self.vectors, self.num_vectors)?;

        self.centroids = kmeans
            .centroids()
            .iter()
            .flat_map(|c: &Vec<f32>| c.iter().copied())
            .collect();

        // Build HNSW coarse quantizer over centroids
        #[cfg(feature = "hnsw")]
        {
            let nc = self.centroids.len() / self.dimension;
            let mut hnsw = crate::hnsw::HNSWIndex::builder(self.dimension)
                .m(16)
                .ef_construction(200)
                .auto_normalize(true)
                .build()?;
            for i in 0..nc {
                let centroid = self.get_centroid(i);
                hnsw.add_slice(i as u32, centroid)?;
            }
            hnsw.build()?;
            self.coarse_quantizer = Some(hnsw);
        }

        // Stage 2: assign vectors to clusters
        let assignments = kmeans.assign_clusters(&self.vectors, self.num_vectors);

        // Build per-cluster vector lists
        let mut cluster_indices: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
        for (vector_idx, &cluster_idx) in assignments.iter().enumerate() {
            cluster_indices[cluster_idx].push(vector_idx as u32);
        }

        // Stage 3: quantize residuals per cluster via qntz's typed edge API.
        // Each cluster centroid plays the role of `parent` in VR; baking
        // `<R*cluster_c, xu_cb>` into the edge lets search form an absolute
        // `||q - v||^2` that is comparable across probed clusters.
        self.clusters = Vec::with_capacity(num_clusters);
        for (cluster_idx, indices) in cluster_indices.into_iter().enumerate() {
            let centroid = self.get_centroid(cluster_idx).to_vec();
            let centroid_rot = self.quantizer.rotate_query(&centroid).map_err(|e| {
                RetrieveError::InvalidParameter(format!("RaBitQ rotate centroid: {e}"))
            })?;
            let mut quantized = Vec::with_capacity(indices.len());

            for &vector_idx in &indices {
                let vec = self.get_vector(vector_idx as usize);
                // R*(v - c) via O(d) subtraction of the pre-rotated views.
                let vec_rot = self
                    .quantizer
                    .rotate_query(vec)
                    .map_err(|e| RetrieveError::InvalidParameter(format!("RaBitQ rotate: {e}")))?;
                let residual_rot: Vec<f32> = vec_rot
                    .iter()
                    .zip(centroid_rot.iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let edge = self
                    .quantizer
                    .quantize_edge_prerotated(&centroid_rot, &residual_rot)
                    .map_err(|e| {
                        RetrieveError::InvalidParameter(format!("RaBitQ quantize: {e}"))
                    })?;
                quantized.push(edge);
            }

            self.clusters.push(Cluster {
                vector_indices: indices,
                quantized,
            });
        }

        self.built = true;
        Ok(())
    }

    /// Drop raw f32 vectors after building to reduce memory usage.
    ///
    /// Calling `compact()` after `build()` drops the raw f32 vectors, reducing
    /// memory by ~4*dim bytes per vector. Search results will use approximate
    /// distances from quantized codes instead of exact reranking.
    ///
    /// # Panics
    ///
    /// Panics if called before `build()`.
    pub fn compact(&mut self) {
        assert!(self.built, "compact() called before build()");
        self.vectors = Vec::new();
        self.compacted = true;
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_ef(query, k, self.params.nprobe)
    }

    /// Search with a custom nprobe value.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        // Normalize query
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };
        let query = query_normalized.as_slice();

        // Find nearest centroids
        let cluster_distances = self.find_nearest_centroids(query, nprobe);

        // Rerank pool scales with nprobe: more probed clusters means noisier
        // RaBitQ approximations need a larger pool to avoid discarding true
        // neighbors. A bounded max-heap avoids the O(N log N) sort cost.
        let rerank_size = (k * 10).max(k * nprobe).max(64);

        // Pre-rotate query once (O(d^2)), then use O(d) prerotated distance per candidate.
        let rotated_query = self
            .quantizer
            .rotate_query(query)
            .map_err(|e| RetrieveError::InvalidParameter(format!("rotate query: {e}")))?;

        // Phase 1: approximate distances via RaBitQ with bounded shortlist.
        // Use a max-heap so we can evict the worst candidate in O(log n).
        let mut heap: std::collections::BinaryHeap<(FloatOrd, u32)> =
            std::collections::BinaryHeap::with_capacity(rerank_size + 1);

        for (cluster_idx, _coarse_dist) in &cluster_distances {
            let cluster = &self.clusters[*cluster_idx];
            if cluster.vector_indices.is_empty() {
                continue;
            }

            // `rotated_query` is R*q (quantizer has no centroid set, so
            // rotate_query does not subtract one). edge_distance_term_prerotated
            // expects exactly that. Add ||q - cluster_c||^2 to compose a true
            // absolute distance comparable across probed clusters.
            //
            // `_coarse_dist` from find_nearest_centroids is cosine distance
            // (or HNSW-metric distance), not guaranteed to be L2-squared, so
            // we recompute ||q - c||^2 directly here. One O(d) op per probed
            // cluster is cheap relative to the per-vector edge loop below.
            let cluster_centroid = self.get_centroid(*cluster_idx);
            let q_c_sqr: f32 = query
                .iter()
                .zip(cluster_centroid.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();

            for (i, edge) in cluster.quantized.iter().enumerate() {
                let edge_term =
                    RaBitQQuantizer::edge_distance_term_prerotated(&rotated_query, edge);
                let dist = q_c_sqr + edge_term;
                let vec_idx = cluster.vector_indices[i];

                if heap.len() < rerank_size {
                    heap.push((FloatOrd(dist), vec_idx));
                } else if let Some(&(FloatOrd(worst), _)) = heap.peek() {
                    if dist < worst {
                        heap.pop();
                        heap.push((FloatOrd(dist), vec_idx));
                    }
                }
            }
        }

        // Phase 2: exact reranking with original vectors (skipped when compacted).
        let mut results: Vec<(u32, f32)> = if self.compacted {
            heap.into_iter()
                .map(|(FloatOrd(dist), vec_idx)| (self.doc_ids[vec_idx as usize], dist))
                .collect()
        } else {
            heap.into_iter()
                .map(|(_, vec_idx)| {
                    let vec = self.get_vector(vec_idx as usize);
                    let dist = crate::distance::cosine_distance_normalized(query, vec);
                    (self.doc_ids[vec_idx as usize], dist)
                })
                .collect()
        };

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);
        Ok(results)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Memory usage breakdown for this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let vectors_bytes = self.vectors.len() * std::mem::size_of::<f32>();

        let quantized_bytes: usize = self
            .clusters
            .iter()
            .flat_map(|c| &c.quantized)
            .map(|edge| {
                edge.quantized.binary_codes.len()
                    + edge.quantized.extended_codes.len()
                    + edge.quantized.codes.len() * std::mem::size_of::<u16>()
                    + std::mem::size_of::<f32>() // ip_parent_rot_codes per edge
            })
            .sum();

        let metadata_bytes = self.doc_ids.len() * std::mem::size_of::<u32>()
            + self.centroids.len() * std::mem::size_of::<f32>();

        crate::memory::MemoryReport {
            vectors_bytes,
            graph_bytes: 0,
            quantized_bytes,
            metadata_bytes,
        }
    }

    /// Get vector from flat storage.
    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Get centroid from flat storage.
    #[inline]
    fn get_centroid(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.centroids[start..start + self.dimension]
    }

    /// Find nearest centroids. Uses HNSW coarse quantizer when available.
    fn find_nearest_centroids(&self, query: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        #[cfg(feature = "hnsw")]
        if let Some(ref hnsw) = self.coarse_quantizer {
            let ef = nprobe * 2;
            if let Ok(results) = hnsw.search(query, nprobe, ef.max(nprobe)) {
                return results
                    .into_iter()
                    .map(|(id, d)| (id as usize, d))
                    .collect();
            }
        }

        let num_centroids = self.centroids.len() / self.dimension;
        let mut dists: Vec<(usize, f32)> = (0..num_centroids)
            .map(|idx| {
                let c = self.get_centroid(idx);
                (idx, crate::distance::cosine_distance_normalized(query, c))
            })
            .collect();
        let nprobe = nprobe.min(dists.len());
        if nprobe < dists.len() {
            dists.select_nth_unstable_by(nprobe, |a, b| a.1.total_cmp(&b.1));
            dists.truncate(nprobe);
        }
        dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        dists
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random vectors
        let mut rng = seed;
        (0..n * dim)
            .map(|_| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn build_and_search_basic() {
        let dim = 32;
        let n = 200;
        let data = make_vectors(n, dim, 42);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 8,
            nprobe: 4,
            total_bits: 4,
            seed: 42,
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let query = &data[0..dim]; // search for first vector
        let results = index.search(query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // First vector should be near the top (self-match after normalization)
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "expected doc_id 0 in results: {:?}",
            results
        );
    }

    #[test]
    fn empty_index_returns_error() {
        let params = IVFRaBitQParams::default();
        let mut index = IVFRaBitQIndex::new(32, params).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let params = IVFRaBitQParams::default();
        let mut index = IVFRaBitQIndex::new(32, params).unwrap();
        assert!(index.add(0, vec![1.0; 64]).is_err());
    }

    #[test]
    fn binary_quantization_works() {
        let dim = 64;
        let n = 100;
        let data = make_vectors(n, dim, 99);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4,
            total_bits: 1, // binary only
            seed: 42,
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();

        let results = index.search(&data[0..dim], 3).unwrap();
        assert!(!results.is_empty());
    }

    /// compact() drops raw vectors; search still returns results using quantized distances.
    #[test]
    fn compact_search_works() {
        let dim = 32;
        let n = 200;
        let data = make_vectors(n, dim, 42);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 8,
            nprobe: 4,
            total_bits: 4,
            seed: 42,
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
        index.build().unwrap();
        index.compact();

        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Compact mode uses approximate distances, so ranking may differ.
        // Just verify we get valid doc IDs and non-negative distances.
        for &(id, dist) in &results {
            assert!((id as usize) < n, "doc_id {id} out of range");
            assert!(dist >= 0.0, "negative distance {dist}");
        }
    }

    /// Self-search recall: searching for each vector should return itself.
    #[test]
    fn self_search_recall() {
        let dim = 32;
        let n = 100;
        let data = make_vectors(n, dim, 7);
        let doc_ids: Vec<u32> = (0..n as u32).collect();

        let params = IVFRaBitQParams {
            num_clusters: 4,
            nprobe: 4, // probe all clusters
            total_bits: 4,
            seed: 42,
        };
        let mut index = IVFRaBitQIndex::new(dim, params).unwrap();
        index.add_batch(&doc_ids, &data).unwrap();
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
            recall > 0.7,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }
}
