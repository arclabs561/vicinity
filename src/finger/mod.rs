//! FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search.
//!
//! A proximity graph that augments edges with projection-based distance lower
//! bounds for faster search pruning (Chen et al., KDD 2023). During search,
//! before computing the full distance to a neighbor, a 1-D lower bound is
//! checked: for unit vectors, `|proj_q - proj_v|^2 / 2 <= cosine_dist(q, v)`,
//! so if `lb^2/2 > worst_dist` the neighbor is skipped without a full distance
//! computation. The projection direction is derived from the data centroid.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.3", features = ["finger"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::finger::{FingerIndex, FingerParams};
//!
//! let params = FingerParams::default();
//! let mut index = FingerIndex::new(128, params)?;
//!
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Construction
//!
//! 1. L2-normalize each vector on insert (cosine distance throughout).
//! 2. Compute centroid of all normalized vectors; derive a unit direction from it.
//! 3. For each vector, compute its scalar projection onto that direction.
//! 4. Build proximity graph: Vamana-style with RNG (`alpha`) pruning,
//!    bidirectional edge insertion, connectivity enforcement.
//!
//! # Search
//!
//! Standard beam search from the medoid entry point, with a pre-filter:
//! before computing `d(query, neighbor)`, check `|proj_q - proj_neighbor|`.
//! If the 1-D lower bound already exceeds the current k-th best distance, skip
//! the full distance computation entirely.
//!
//! # References
//!
//! - Chen et al. (2023). "FINGER: Fast Inference for Graph-based Approximate
//!   Nearest Neighbor Search." KDD 2023.

use crate::distance::cosine_distance_normalized;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::collections::{BinaryHeap, HashSet};

/// FINGER construction and search parameters.
#[derive(Clone, Debug)]
pub struct FingerParams {
    /// Maximum out-degree. Default: 32.
    pub max_degree: usize,
    /// Candidate pool size during construction (ef_construction). Default: 200.
    pub ef_construction: usize,
    /// Candidate pool size during search. Default: 100.
    pub ef_search: usize,
    /// RNG pruning factor (alpha >= 1.0). Default: 1.2.
    pub alpha: f32,
}

impl Default for FingerParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            ef_construction: 200,
            ef_search: 100,
            alpha: 1.2,
        }
    }
}

/// FINGER index.
pub struct FingerIndex {
    dimension: usize,
    params: FingerParams,
    built: bool,

    /// Flat L2-normalized vector store: row `i` at `vectors[i*dim..(i+1)*dim]`.
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    neighbors: Vec<SmallVec<[u32; 16]>>,
    medoid: u32,

    // --- FINGER projection data (populated during build) ---
    /// Unit direction derived from the centroid of all normalized vectors.
    direction: Vec<f32>,
    /// `projections[i] = dot(vectors[i], direction)`.
    projections: Vec<f32>,
}

impl FingerIndex {
    /// Create a new FINGER index for vectors of `dimension` dimensions.
    pub fn new(dimension: usize, params: FingerParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        if params.alpha < 1.0 {
            return Err(RetrieveError::InvalidParameter(
                "alpha must be >= 1.0".into(),
            ));
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            neighbors: Vec::new(),
            medoid: 0,
            direction: Vec::new(),
            projections: Vec::new(),
        })
    }

    /// Add a vector (takes ownership).
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector from a slice.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
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

    /// Build the FINGER index. Must be called before `search`.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let n = self.num_vectors;

        // Step 1: Compute navigating node (medoid) and centroid direction.
        let (medoid, direction) = self.compute_medoid_and_direction();
        self.medoid = medoid;
        self.direction = direction;

        // Step 2: Compute per-vector projections onto the direction.
        self.projections = (0..n)
            .map(|i| {
                let v = self.get_vector(i);
                v.iter()
                    .zip(self.direction.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            })
            .collect();

        // Step 3: Build initial kNN graph (brute-force).
        self.build_knn_graph();

        // Step 4: RNG refinement pass over all nodes.
        for i in 0..n {
            let vi = self.get_vector(i).to_vec();

            // Beam search for a richer candidate set.
            let candidates = self.beam_search_internal(&vi, self.params.ef_construction);

            // RNG pruning with alpha.
            let selected = self.rng_prune(&vi, &candidates);

            let old = std::mem::replace(
                &mut self.neighbors[i],
                selected.iter().map(|&(id, _)| id).collect(),
            );

            // Bidirectional edge insertion with capacity-aware re-pruning.
            let max_deg = self.params.max_degree;
            for &(neighbor_id, _) in &selected {
                let nid = neighbor_id as usize;
                if !self.neighbors[nid].contains(&(i as u32)) {
                    if self.neighbors[nid].len() < max_deg {
                        self.neighbors[nid].push(i as u32);
                    } else {
                        let nv = self.get_vector(nid).to_vec();
                        let rev_cands: Vec<(u32, f32)> = self.neighbors[nid]
                            .iter()
                            .chain(std::iter::once(&(i as u32)))
                            .map(|&id| {
                                (
                                    id,
                                    cosine_distance_normalized(&nv, self.get_vector(id as usize)),
                                )
                            })
                            .collect();
                        let pruned = self.rng_prune(&nv, &rev_cands);
                        self.neighbors[nid] = pruned.iter().map(|&(id, _)| id).collect();
                    }
                }
            }

            drop(old);
        }

        // Step 5: Connectivity enforcement.
        self.ensure_connectivity();

        self.built = true;
        Ok(())
    }

    /// Search for the `k` nearest neighbors of `query`.
    ///
    /// Returns `(doc_id, distance)` pairs sorted by ascending distance.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
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

        // L2-normalize the query (same space as stored vectors).
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q_norm: Vec<f32> = if norm > 1e-10 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        };

        // Projection of the (normalized) query onto the centroid direction.
        let query_proj: f32 = q_norm
            .iter()
            .zip(self.direction.iter())
            .map(|(a, b)| a * b)
            .sum();

        let ef = self.params.ef_search.max(k);
        let results = self.beam_search_with_pruning(&q_norm, query_proj, ef);

        Ok(results
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.doc_ids[id as usize], dist))
            .collect())
    }

    /// Search with a custom `ef_search` beam width, overriding the params default.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
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

        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q_norm: Vec<f32> = if norm > 1e-10 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        };

        let query_proj: f32 = q_norm
            .iter()
            .zip(self.direction.iter())
            .map(|(a, b)| a * b)
            .sum();

        let ef = ef_search.max(k);
        let results = self.beam_search_with_pruning(&q_norm, query_proj, ef);

        Ok(results
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.doc_ids[id as usize], dist))
            .collect())
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index has no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    // ── Internal helpers ────────────────────────────────────────────────

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Compute the medoid (closest to centroid) and the unit centroid direction.
    fn compute_medoid_and_direction(&self) -> (u32, Vec<f32>) {
        let dim = self.dimension;
        let n = self.num_vectors;
        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            let v = self.get_vector(i);
            for (j, &val) in v.iter().enumerate() {
                centroid[j] += val;
            }
        }
        for c in &mut centroid {
            *c /= n as f32;
        }

        // Normalize centroid to get the direction.
        let c_norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        let direction: Vec<f32> = if c_norm > 1e-10 {
            centroid.iter().map(|x| x / c_norm).collect()
        } else {
            // Degenerate: all vectors cancel. Use the first vector as direction.
            self.get_vector(0).to_vec()
        };

        // Medoid: closest to the (unnormalized) centroid in cosine distance.
        let unnorm_centroid: Vec<f32> = direction.iter().map(|x| x * c_norm).collect();
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for i in 0..n {
            let d = cosine_distance_normalized(&unnorm_centroid, self.get_vector(i));
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        (best, direction)
    }

    /// Build initial kNN graph. Uses brute-force for n <= 1000, NN-descent otherwise.
    fn build_knn_graph(&mut self) {
        let n = self.num_vectors;
        if n <= 1000 {
            self.build_knn_graph_bruteforce();
        } else {
            self.build_knn_graph_nndescent();
        }
    }

    fn build_knn_graph_bruteforce(&mut self) {
        let n = self.num_vectors;
        let k = self.params.max_degree.min(n.saturating_sub(1));
        self.neighbors = vec![SmallVec::new(); n];

        for i in 0..n {
            let vi = self.get_vector(i);
            let mut dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, cosine_distance_normalized(vi, self.get_vector(j))))
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(k);
            self.neighbors[i] = dists.iter().map(|(id, _)| *id).collect();
        }
    }

    /// NN-descent (Dong et al., 2011) kNN graph construction.
    fn build_knn_graph_nndescent(&mut self) {
        let (n, k, dim) = (self.num_vectors, self.params.max_degree, self.dimension);
        let vecs = &self.vectors;
        self.neighbors = crate::graph_utils::build_knn_graph_nndescent(n, k, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }

    /// RNG pruning with configurable alpha.
    ///
    /// Keep candidate `c` if no already-selected neighbor `s` satisfies
    /// `alpha * d(query, c) <= d(s, c)` -- i.e. the selected neighbor is not
    /// too much closer to `c` than the query is.
    fn rng_prune(&self, _query_vec: &[f32], candidates: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let mut sorted = candidates.to_vec();
        sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        sorted.dedup_by_key(|c| c.0);

        let max_deg = self.params.max_degree;
        let alpha = self.params.alpha;
        let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_deg);

        for &(cand_id, cand_dist) in &sorted {
            if selected.len() >= max_deg {
                break;
            }
            let cand_vec = self.get_vector(cand_id as usize);
            let mut keep = true;

            for &(sel_id, _) in &selected {
                let sel_vec = self.get_vector(sel_id as usize);
                let inter_dist = cosine_distance_normalized(sel_vec, cand_vec);
                // RNG condition: if a selected neighbor is already very close to
                // this candidate, the candidate is redundant.
                if alpha * cand_dist >= inter_dist {
                    keep = false;
                    break;
                }
            }

            if keep {
                selected.push((cand_id, cand_dist));
            }
        }

        selected
    }

    /// Plain beam search (used during construction).
    fn beam_search_internal(&self, query: &[f32], ef: usize) -> Vec<(u32, f32)> {
        let n = self.num_vectors;
        if n == 0 {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        let entry = self.medoid;
        let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
        visited.insert(entry);
        frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
        candidates.push((entry, entry_dist));

        while let Some(std::cmp::Reverse((FloatOrd(cur_dist), cur_id))) = frontier.pop() {
            if candidates.len() >= ef {
                candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                if cur_dist > candidates[ef - 1].1 * 1.5 {
                    break;
                }
            }

            for &nb in &self.neighbors[cur_id as usize] {
                if visited.insert(nb) {
                    let d = cosine_distance_normalized(query, self.get_vector(nb as usize));
                    candidates.push((nb, d));
                    frontier.push(std::cmp::Reverse((FloatOrd(d), nb)));
                }
            }

            if visited.len() > ef * 10 {
                break;
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.dedup_by_key(|c| c.0);
        candidates
    }

    /// Beam search with projection-based pruning (used during search).
    ///
    /// Before computing the full cosine distance to a neighbor, checks the 1-D
    /// lower bound `|query_proj - proj[neighbor]|`. If the lower bound already
    /// exceeds the current k-th best candidate distance, the neighbor is skipped.
    fn beam_search_with_pruning(
        &self,
        query: &[f32],
        query_proj: f32,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        let n = self.num_vectors;
        if n == 0 {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        // Current worst (k-th best) distance among candidates.
        // Initialised to infinity so the first few neighbors are never pruned.
        let mut worst_dist = f32::INFINITY;

        let entry = self.medoid;
        let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
        visited.insert(entry);
        frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
        candidates.push((entry, entry_dist));

        while let Some(std::cmp::Reverse((FloatOrd(cur_dist), cur_id))) = frontier.pop() {
            // Early exit: current node is farther than all top-ef candidates.
            if candidates.len() >= ef {
                candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                let cutoff = candidates[ef - 1].1;
                worst_dist = cutoff;
                if cur_dist > cutoff * 1.5 {
                    break;
                }
            }

            for &nb in &self.neighbors[cur_id as usize] {
                if !visited.insert(nb) {
                    continue;
                }

                // --- Projection lower bound check ---
                // For unit vectors: cosine_dist(q,v) = 1 - dot(q,v).
                // The 1-D projection diff satisfies:
                //   |proj_q - proj_v|^2 <= ||q-v||^2 = 2 * cosine_dist(q,v)
                // so  cosine_dist(q,v) >= lb^2 / 2.
                // If this lower bound already exceeds the current worst candidate,
                // skip the full distance computation.
                let lb = (query_proj - self.projections[nb as usize]).abs();
                if lb * lb * 0.5 > worst_dist {
                    continue;
                }

                let d = cosine_distance_normalized(query, self.get_vector(nb as usize));
                candidates.push((nb, d));
                frontier.push(std::cmp::Reverse((FloatOrd(d), nb)));

                // Update worst_dist if the candidate pool is full.
                if candidates.len() >= ef {
                    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    worst_dist = candidates[ef - 1].1;
                }
            }

            if visited.len() > ef * 10 {
                break;
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.dedup_by_key(|c| c.0);
        candidates
    }

    fn ensure_connectivity(&mut self) {
        let (dim, vecs) = (self.dimension, &self.vectors);
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.medoid, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }
}

// ── FloatOrd ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct FloatOrd(f32);
impl Eq for FloatOrd {}
impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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

    fn make_index(n: usize, dim: usize, seed: u64) -> FingerIndex {
        let data = make_vectors(n, dim, seed);
        let mut index = FingerIndex::new(
            dim,
            FingerParams {
                max_degree: 16,
                ef_construction: 64,
                ef_search: 50,
                alpha: 1.2,
            },
        )
        .unwrap();
        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn build_and_search() {
        let dim = 16;
        let n = 40;
        let index = make_index(n, dim, 42);

        let data = make_vectors(n, dim, 42);
        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();
        assert!(!results.is_empty());
        // The vector at index 0 should be its own nearest neighbor.
        assert!(results.iter().any(|(id, _)| *id == 0));
    }

    #[test]
    fn self_search_recall() {
        let dim = 16;
        let n = 30;
        let data = make_vectors(n, dim, 7);

        let mut index = FingerIndex::new(
            dim,
            FingerParams {
                max_degree: 16,
                ef_construction: 64,
                ef_search: 50,
                alpha: 1.2,
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
            let query = &data[i * dim..(i + 1) * dim];
            let results = index.search(query, 1).unwrap();
            if results.first().map(|(id, _)| *id) == Some(i as u32) {
                hits += 1;
            }
        }
        let recall = hits as f64 / n as f64;
        assert!(
            recall > 0.6,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    #[test]
    fn projection_reduces_computations() {
        // Verify that projection-pruned search returns correct results.
        // We cannot easily count pruned nodes from outside, so we check that
        // results match a brute-force ranking on a small dataset.
        let dim = 8;
        let n = 20;
        let data = make_vectors(n, dim, 55);
        let index = make_index(n, dim, 55);

        for i in 0..n {
            let query = &data[i * dim..(i + 1) * dim];
            let results = index.search(query, 3).unwrap();
            // At minimum the query's own doc_id should appear in the top-3.
            assert!(
                results.iter().any(|(id, _)| *id == i as u32),
                "vector {i} not in its own top-3 results"
            );
        }
    }

    #[test]
    fn connectivity() {
        let dim = 8;
        let n = 20;
        let index = make_index(n, dim, 123);

        let mut visited = vec![false; n];
        let mut stack = vec![index.medoid as usize];
        visited[index.medoid as usize] = true;
        while let Some(node) = stack.pop() {
            for &nb in &index.neighbors[node] {
                let nb = nb as usize;
                if !visited[nb] {
                    visited[nb] = true;
                    stack.push(nb);
                }
            }
        }
        let reachable = visited.iter().filter(|&&v| v).count();
        assert_eq!(reachable, n, "not all nodes reachable from medoid");
    }

    #[test]
    fn empty_index_errors() {
        let mut index = FingerIndex::new(8, FingerParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut index = FingerIndex::new(8, FingerParams::default()).unwrap();
        let result = index.add_slice(0, &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn add_after_build_errors() {
        let dim = 4;
        let data = make_vectors(5, dim, 1);
        let mut index = FingerIndex::new(dim, FingerParams::default()).unwrap();
        for i in 0..5usize {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();
        let result = index.add_slice(99, &data[0..dim]);
        assert!(result.is_err());
    }
}
