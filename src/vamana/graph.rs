//! Vamana graph structure and core types.

use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::path::Path;

const VAMANA_FORMAT_VERSION: u32 = 1;
const VAMANA_NEIGHBORS_MAGIC: &[u8; 8] = b"VAMANAG1";

#[cfg(feature = "vamana")]
/// Vamana parameters controlling graph structure and search behavior.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VamanaParams {
    /// Maximum out-degree per node (typically 64-128, higher for SSD serving)
    pub max_degree: usize,

    /// Relaxation factor for RRND (typically 1.3-1.5)
    /// Higher alpha = less pruning, larger graphs
    pub alpha: f32,

    /// Search width during construction (typically 200-400)
    pub ef_construction: usize,

    /// Default search width during query (typically 50-200)
    pub ef_search: usize,

    /// Optional RNG seed for reproducible construction.
    /// When `None` (default), uses thread-local RNG.
    pub seed: Option<u64>,
}

#[cfg(feature = "vamana")]
impl Default for VamanaParams {
    fn default() -> Self {
        Self {
            max_degree: 64,
            alpha: 1.3,
            ef_construction: 200,
            ef_search: 50,
            seed: None,
        }
    }
}

#[cfg(feature = "vamana")]
/// Vamana index for approximate nearest neighbor search.
///
/// Uses two-pass construction (RND then RRND) for high-quality graph
/// structure.
///
/// **Cosine distance only.** Input vectors are L2-normalized in
/// [`Self::add`]; pre-normalization is optional. The
/// [`crate::distance::DistanceMetric`] enum is not honored — for L2 or
/// inner product on a Vamana-style graph, a different index is needed.
pub struct VamanaIndex {
    /// Vector dimension
    pub(crate) dimension: usize,

    /// Vectors stored in Structure of Arrays (SoA) layout
    pub(crate) vectors: Vec<f32>,

    /// Neighbor lists for each vector
    pub(crate) neighbors: Vec<SmallVec<[u32; 32]>>,

    /// Parameters
    pub(crate) params: VamanaParams,

    /// Number of vectors added
    pub(crate) num_vectors: usize,

    /// External doc IDs (maps internal index -> caller-provided ID)
    pub(crate) doc_ids: Vec<u32>,

    /// Whether index has been built
    built: bool,

    /// Medoid (closest point to centroid), used as search entry point.
    /// Computed during build.
    pub(crate) medoid: u32,
}

#[derive(Deserialize, Serialize)]
struct VamanaSnapshot {
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: VamanaParams,
    medoid: u32,
}

#[cfg(feature = "vamana")]
impl VamanaIndex {
    /// Create a new Vamana index.
    pub fn new(dimension: usize, params: VamanaParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        Ok(Self {
            dimension,
            vectors: Vec::new(),
            neighbors: Vec::new(),
            params,
            num_vectors: 0,
            doc_ids: Vec::new(),
            built: false,
            medoid: 0,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after build".into(),
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
            self.vectors.extend_from_slice(&vector);
        }
        self.neighbors.push(SmallVec::new());
        self.doc_ids.push(id);
        self.num_vectors += 1;

        Ok(())
    }

    /// Build the index (two-pass construction).
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "index already built".into(),
            ));
        }

        // Two-pass construction: RRND + RND
        super::construction::construct_graph(self)?;
        self.built = true;
        self.reorder_for_locality();

        Ok(())
    }

    /// Save a built Vamana index to a directory.
    ///
    /// The saved format stores the post-build, locality-reordered graph state.
    /// Loading restores that graph directly; it does not rebuild.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt Vamana index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let snapshot = VamanaSnapshot {
            version: VAMANA_FORMAT_VERSION,
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            params: self.params.clone(),
            medoid: self.medoid,
        };
        crate::graph_snapshot::write_json_atomic(&output_dir.join("manifest.json"), &snapshot)?;
        crate::graph_snapshot::write_f32_atomic(&output_dir.join("vectors.bin"), &self.vectors)?;
        crate::graph_snapshot::write_u32_atomic(&output_dir.join("doc_ids.bin"), &self.doc_ids)?;
        crate::graph_snapshot::write_neighbors_atomic(
            &output_dir.join("neighbors.bin"),
            VAMANA_NEIGHBORS_MAGIC,
            &self.neighbors,
        )?;
        Ok(())
    }

    /// Load a Vamana index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let snapshot: VamanaSnapshot =
            crate::graph_snapshot::read_json(&input_dir.join("manifest.json"))?;
        if snapshot.version != VAMANA_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported Vamana format version {}",
                snapshot.version
            )));
        }
        let vectors = crate::graph_snapshot::read_f32_exact(
            &input_dir.join("vectors.bin"),
            snapshot.num_vectors * snapshot.dimension,
        )?;
        let doc_ids = crate::graph_snapshot::read_u32_exact(
            &input_dir.join("doc_ids.bin"),
            snapshot.num_vectors,
        )?;
        let neighbors = crate::graph_snapshot::read_neighbors(
            &input_dir.join("neighbors.bin"),
            VAMANA_NEIGHBORS_MAGIC,
            snapshot.num_vectors,
        )?;
        crate::graph_snapshot::validate_graph_shape(
            "Vamana",
            snapshot.dimension,
            snapshot.num_vectors,
            &vectors,
            &doc_ids,
            &neighbors,
            Some(snapshot.medoid),
        )?;
        Ok(Self {
            dimension: snapshot.dimension,
            vectors,
            neighbors,
            params: snapshot.params,
            num_vectors: snapshot.num_vectors,
            doc_ids,
            built: true,
            medoid: snapshot.medoid,
        })
    }

    /// Build using parallel batched construction (requires `parallel` feature).
    ///
    /// Same two-pass structure as [`build`](Self::build) but parallelizes the
    /// neighbor search phase within each pass using rayon.
    #[cfg(feature = "parallel")]
    pub fn build_parallel(&mut self, batch_size: usize) -> Result<(), RetrieveError> {
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "index already built".into(),
            ));
        }
        super::construction::construct_graph_parallel(self, batch_size)?;
        self.built = true;
        self.reorder_for_locality();
        Ok(())
    }

    /// Memory usage breakdown for this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        crate::memory::MemoryReport {
            vectors_bytes: self.vectors.len() * std::mem::size_of::<f32>(),
            graph_bytes: crate::memory::smallvec_u32_bytes(&self.neighbors),
            quantized_bytes: 0,
            metadata_bytes: self.doc_ids.len() * std::mem::size_of::<u32>(),
        }
    }

    /// BFS-order graph reordering for cache-friendly traversal.
    fn reorder_for_locality(&mut self) {
        if self.num_vectors <= 1 {
            return;
        }
        let n = self.num_vectors;
        let dim = self.dimension;
        let ep = self.medoid as usize;

        // BFS from medoid
        let mut new_order: Vec<u32> = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut queue = std::collections::VecDeque::with_capacity(n);
        queue.push_back(ep);
        visited[ep] = true;

        while let Some(node) = queue.pop_front() {
            new_order.push(node as u32);
            for &nb in &self.neighbors[node] {
                let nb = nb as usize;
                if nb < n && !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if !visited[i] {
                new_order.push(i as u32);
            }
        }

        // Build permutation
        let mut old_to_new = vec![0u32; n];
        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            old_to_new[old_idx as usize] = new_idx as u32;
        }

        // Permute vectors
        let mut new_vectors = vec![0.0f32; self.vectors.len()];
        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            let src = old_idx as usize * dim;
            let dst = new_idx * dim;
            new_vectors[dst..dst + dim].copy_from_slice(&self.vectors[src..src + dim]);
        }
        self.vectors = new_vectors;

        // Permute doc_ids
        let mut new_doc_ids = vec![0u32; n];
        for (new_idx, &old_idx) in new_order.iter().enumerate() {
            new_doc_ids[new_idx] = self.doc_ids[old_idx as usize];
        }
        self.doc_ids = new_doc_ids;

        // Permute and remap neighbor lists
        let mut new_neighbors = vec![SmallVec::new(); n];
        for (old_idx, nbs) in self.neighbors.iter().enumerate() {
            if old_idx < n {
                let new_idx = old_to_new[old_idx] as usize;
                new_neighbors[new_idx] = nbs.iter().map(|&nb| old_to_new[nb as usize]).collect();
            }
        }
        self.neighbors = new_neighbors;

        // Update medoid
        self.medoid = old_to_new[ep];
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }

        // Normalize query for cosine distance
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };
        super::search::search(self, &query_normalized, k, ef)
    }

    /// Search with a custom distance function.
    ///
    /// The closure receives `(query, internal_node_id)` and returns a distance.
    /// Enables ADSampling and other asymmetric distance schemes.
    pub fn search_with_distance<F: Fn(&[f32], u32) -> f32>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        dist_fn: &F,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        super::search::search_with_distance(self, query, k, ef, dist_fn)
    }

    /// Get vector by index.
    #[inline]
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

#[cfg(all(test, feature = "vamana"))]
mod tests {
    use super::*;
    use crate::distance;

    #[test]
    fn test_vamana_add() {
        let params = VamanaParams::default();
        let mut index = VamanaIndex::new(128, params).unwrap();

        let vector = vec![0.1; 128];
        index
            .add(0, vector)
            .expect("add must succeed on a fresh index");
        assert_eq!(index.num_vectors, 1);
    }

    /// Generate `n` random normalized vectors of given dimension using a simple LCG.
    fn generate_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let raw: Vec<f32> = (0..dim)
                    .map(|_| {
                        // Simple LCG for reproducibility without extra deps
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        // Map to [-1, 1]
                        ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
                    })
                    .collect();
                distance::normalize(&raw)
            })
            .collect()
    }

    #[test]
    fn test_vamana_build_does_not_panic() {
        let dim = 32;
        let n = 60;
        let vectors = generate_normalized_vectors(n, dim, 42);

        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 40,
            ef_search: 20,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        // Must not panic
        index.build().unwrap();

        // Medoid should be within valid range
        assert!((index.medoid as usize) < n);
    }

    #[test]
    fn test_vamana_search_self_query() {
        let dim = 32;
        let n = 80;
        let vectors = generate_normalized_vectors(n, dim, 99);

        let params = VamanaParams {
            max_degree: 32,
            alpha: 1.5,
            ef_construction: 100,
            ef_search: 80,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // For each of a sample of vectors, the self-query should appear in the
        // top-k results with distance close to zero.
        let k = 5;
        let ef = 80;
        let sample_indices = [0, 1, n / 2, n - 1];
        for &idx in &sample_indices {
            let results = index.search(&vectors[idx], k, ef).unwrap();
            assert!(
                !results.is_empty(),
                "search returned empty results for query {}",
                idx
            );
            // After graph reordering, internal IDs change. Check for
            // near-zero distance (self-retrieval) regardless of ID.
            let found = results.iter().any(|&(_, dist)| dist < 1e-4);
            assert!(
                found,
                "self-query for vector {} not found in top-{} results: {:?}",
                idx, k, results
            );
        }
    }

    #[test]
    fn test_save_load_roundtrip_preserves_search() {
        let dim = 16;
        let n = 80;
        let vectors = generate_normalized_vectors(n, dim, 99);
        let params = VamanaParams {
            max_degree: 16,
            ef_construction: 60,
            ef_search: 40,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(3000 + i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();
        let query = &vectors[7];
        let before = index.search(query, 8, 40).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = VamanaIndex::load_from_dir(dir.path()).unwrap();
        assert_eq!(loaded.search(query, 8, 40).unwrap(), before);
    }

    #[test]
    fn test_vamana_search_deterministic() {
        let dim = 32;
        let n = 60;
        let vectors = generate_normalized_vectors(n, dim, 77);

        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 40,
            ef_search: 30,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // Same query should return same results (medoid entry point is deterministic)
        let query = &vectors[5];
        let r1 = index.search(query, 5, 30).unwrap();
        let r2 = index.search(query, 5, 30).unwrap();
        assert_eq!(
            r1, r2,
            "search should be deterministic with medoid entry point"
        );
    }

    #[test]
    fn test_vamana_max_degree_enforced() {
        let dim = 16;
        let n = 100;
        let vectors = generate_normalized_vectors(n, dim, 42);

        let max_degree = 16;
        let params = VamanaParams {
            max_degree,
            alpha: 1.3,
            ef_construction: 60,
            ef_search: 30,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // No node should exceed max_degree neighbors
        for (node_id, neighbors) in index.neighbors.iter().enumerate() {
            assert!(
                neighbors.len() <= max_degree,
                "Node {} has {} neighbors, max_degree is {}",
                node_id,
                neighbors.len(),
                max_degree
            );
        }
    }

    #[test]
    fn test_vamana_neighbor_ids_in_bounds() {
        let dim = 16;
        let n = 80;
        let vectors = generate_normalized_vectors(n, dim, 99);

        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 60,
            ef_search: 30,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        for (node_id, neighbors) in index.neighbors.iter().enumerate() {
            for &nbr in neighbors.iter() {
                assert!(
                    (nbr as usize) < n,
                    "Node {} has out-of-bounds neighbor {}",
                    node_id,
                    nbr
                );
            }
        }
    }

    /// Recall regression test: greedy beam search construction must produce
    /// recall@10 > 70% on synthetic data.
    ///
    /// The prior BFS-based construction gave ~3–48% recall (measured on GloVe-25).
    /// This test catches a regression to that incorrect algorithm.
    #[test]
    fn test_construction_recall_regression() {
        let n = 400;
        let dim = 16;
        let k = 10;
        let ef = 200;
        let vecs = generate_normalized_vectors(n, dim, 123);

        let params = VamanaParams {
            max_degree: 32,
            alpha: 1.3,
            ef_construction: 200,
            ef_search: ef,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vecs.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // Compute recall against brute-force for 20 evenly spaced query vectors.
        let step = (n / 20).max(1);
        let mut total_recall = 0.0f64;
        let mut num_queries = 0usize;

        for qi in (0..n).step_by(step) {
            let query = &vecs[qi];

            let mut all_dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != qi)
                .map(|j| {
                    (
                        j as u32,
                        distance::cosine_distance_normalized(query, &vecs[j]),
                    )
                })
                .collect();
            all_dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            // Use distance-based recall: a result is "correct" if its distance
            // matches any of the true k-NN distances within tolerance.
            // This is robust to graph reordering (which changes internal IDs).
            let true_dists: Vec<f32> = all_dists.iter().take(k).map(|&(_, d)| d).collect();
            let worst_true = true_dists.last().copied().unwrap_or(f32::INFINITY);

            let results = index.search(query, k, ef).unwrap();
            let hits = results
                .iter()
                .filter(|&&(_, d)| d <= worst_true + 1e-5)
                .count();

            total_recall += hits as f64 / k as f64;
            num_queries += 1;
        }

        let mean_recall = total_recall / num_queries as f64;
        assert!(
            mean_recall >= 0.70,
            "Recall@{k} = {:.1}% < 70% (construction quality regression; BFS construction gave ~40%)",
            mean_recall * 100.0,
        );
    }

    #[test]
    fn test_vamana_graph_connected() {
        // After construction, every node should be reachable from the medoid
        // via BFS through the neighbor graph.
        let dim = 16;
        let n = 50;
        let vectors = generate_normalized_vectors(n, dim, 55);

        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.5,
            ef_construction: 100,
            ef_search: 50,
            seed: None,
            ..VamanaParams::default()
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // BFS from medoid
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(index.medoid);
        visited.insert(index.medoid);

        while let Some(node) = queue.pop_front() {
            for &nbr in index.neighbors[node as usize].iter() {
                if visited.insert(nbr) {
                    queue.push_back(nbr);
                }
            }
        }

        assert_eq!(
            visited.len(),
            n,
            "Only {} of {} nodes reachable from medoid (graph disconnected)",
            visited.len(),
            n
        );
    }

    /// Search must return caller-provided doc_ids, not internal indices.
    /// Regression test: BFS graph reorder permuted internal IDs without
    /// mapping back to doc_ids, causing 0% recall on all benchmarks.
    #[test]
    fn test_search_returns_doc_ids_not_internal_indices() {
        let dim = 16;
        let n = 200;
        let vectors = generate_normalized_vectors(n, dim, 12345);
        let params = VamanaParams::default();
        let mut index = VamanaIndex::new(dim, params).unwrap();

        // Use non-trivial doc_ids (offset by 1000) so internal != external
        for (i, v) in vectors.iter().enumerate() {
            index.add((i + 1000) as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        let results = index.search(&vectors[0], 10, 50).unwrap();
        for (doc_id, _dist) in &results {
            assert!(
                *doc_id >= 1000 && *doc_id < (1000 + n as u32),
                "search returned id {} which is outside the doc_id range [1000, {})",
                doc_id,
                1000 + n as u32
            );
        }
    }

    /// Self-query must find itself (doc_id match, not just distance ~0).
    #[test]
    fn test_self_query_returns_correct_doc_id() {
        let dim = 16;
        let n = 100;
        let vectors = generate_normalized_vectors(n, dim, 54321);
        let params = VamanaParams::default();
        let mut index = VamanaIndex::new(dim, params).unwrap();

        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // Query with each vector -- it should appear in its own results
        for i in 0..n.min(20) {
            let results = index.search(&vectors[i], 10, 50).unwrap();
            let ids: Vec<u32> = results.iter().map(|r| r.0).collect();
            assert!(
                ids.contains(&(i as u32)),
                "vector {} not found in its own search results: {:?}",
                i,
                ids
            );
        }
    }
}
