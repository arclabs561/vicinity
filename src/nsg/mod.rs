//! NSG: Navigating Spreading-out Graph.
//!
//! A single-layer proximity graph that combines MRNG (Monotone Relative
//! Neighborhood Graph) pruning with a navigating node (medoid) entry point.
//! NSG fills the gap between HNSW (hierarchical, higher memory) and
//! brute-force kNN graphs (dense, slower search) on the Pareto curve.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.5", features = ["nsg"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::nsg::{NsgIndex, NsgParams};
//!
//! let params = NsgParams::default();
//! let mut index = NsgIndex::new(128, params)?;
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
//! 1. Build initial kNN graph (brute-force for small n, NN-descent-like for large)
//! 2. Compute navigating node (medoid = closest to centroid)
//! 3. MRNG pruning: for each node, beam-search for candidates, then occlusion
//!    prune (same as HNSW's RND diversification with alpha=1.0)
//! 4. Bidirectional edge insertion with capacity-aware pruning
//! 5. Connectivity enforcement: DFS from navigating node, connect unreachable
//!
//! # References
//!
//! - Fu, Xiang, Wang, Huang (2019). "Fast Approximate Nearest Neighbor Search
//!   With The Navigating Spreading-out Graph." PVLDB 12(5).

use crate::distance::cosine_distance_normalized;
use crate::distance::FloatOrd;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::collections::BinaryHeap;

/// NSG parameters.
#[derive(Clone, Debug)]
pub struct NsgParams {
    /// Maximum out-degree (R in the paper). Default: 32.
    pub max_degree: usize,
    /// Construction pool size (L in the paper). Default: 64.
    pub pool_size: usize,
    /// Initial kNN graph degree (K in the paper). Default: 32.
    pub knn_degree: usize,
    /// Search ef parameter. Default: 100.
    pub ef_search: usize,
}

impl Default for NsgParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            pool_size: 64,
            knn_degree: 32,
            ef_search: 100,
        }
    }
}

/// NSG index.
pub struct NsgIndex {
    dimension: usize,
    params: NsgParams,
    built: bool,

    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    neighbors: Vec<SmallVec<[u32; 16]>>,
    medoid: u32,
}

impl NsgIndex {
    /// Create a new NSG index.
    pub fn new(dimension: usize, params: NsgParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
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
        })
    }

    /// Add a vector.
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

    /// Build the NSG index.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let n = self.num_vectors;

        // Step 1: Compute navigating node (medoid)
        self.medoid = self.compute_medoid();

        // Step 2: Build initial kNN graph
        self.build_knn_graph();

        // Step 3: MRNG refinement -- for each node, beam search for candidates,
        // then occlusion prune (RND with alpha=1.0)
        for i in 0..n {
            let vi = self.get_vector(i).to_vec();

            // Beam search from medoid to find candidate neighbors
            let candidates = self.beam_search(&vi, self.params.pool_size);

            // MRNG pruning (same as RND: keep candidate if no existing neighbor
            // is closer to it than the candidate is to the query node)
            let selected = self.mrng_prune(&vi, &candidates);

            // Inter-insert: add bidirectional edges
            let old_neighbors = std::mem::replace(
                &mut self.neighbors[i],
                selected.iter().map(|&(id, _)| id).collect(),
            );

            // Add reverse edges
            let max_deg = self.params.max_degree;
            for &(neighbor_id, _) in &selected {
                let nid = neighbor_id as usize;
                if !self.neighbors[nid].contains(&(i as u32)) {
                    if self.neighbors[nid].len() < max_deg {
                        self.neighbors[nid].push(i as u32);
                    } else {
                        // Re-prune reverse neighbor list
                        let nv = self.get_vector(nid).to_vec();
                        let rev_cands: Vec<(u32, f32)> = self.neighbors[nid]
                            .iter()
                            .chain(std::iter::once(&(i as u32)))
                            .map(|&id| {
                                let d =
                                    cosine_distance_normalized(&nv, self.get_vector(id as usize));
                                (id, d)
                            })
                            .collect();
                        let pruned = self.mrng_prune(&nv, &rev_cands);
                        self.neighbors[nid] = pruned.iter().map(|&(id, _)| id).collect();
                    }
                }
            }

            // Drop old_neighbors (was only needed to avoid borrow conflict)
            drop(old_neighbors);
        }

        // Step 4: Connectivity enforcement
        self.ensure_connectivity();

        self.built = true;
        Ok(())
    }

    /// Search for k nearest neighbors.
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

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };

        let results = self.beam_search(&query_normalized, self.params.ef_search.max(k));

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

        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };

        let results = self.beam_search(&query_normalized, ef_search.max(k));

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

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    // ── Internal ───────────────────────────────────────────────────────

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    fn compute_medoid(&self) -> u32 {
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
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for i in 0..n {
            let d = cosine_distance_normalized(&centroid, self.get_vector(i));
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
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
        let k = self.params.knn_degree.min(n - 1);
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
    ///
    fn build_knn_graph_nndescent(&mut self) {
        let (n, k, dim) = (self.num_vectors, self.params.knn_degree, self.dimension);
        let vecs = &self.vectors;
        self.neighbors = crate::graph_utils::build_knn_graph_nndescent(n, k, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }

    /// MRNG pruning (identical to RND with alpha=1.0).
    /// Keep candidate c if no already-selected neighbor s is closer to c than c is to query.
    fn mrng_prune(&self, _query_vec: &[f32], candidates: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
        sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        sorted.dedup_by_key(|c| c.0);

        let max_deg = self.params.max_degree;
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
                // RND condition (alpha=1.0): candidate must be closer to query
                // than to any already-selected neighbor
                if cand_dist >= inter_dist {
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

    /// Beam search from medoid (standard NSG greedy search).
    ///
    /// Maintains a bounded result set of size `ef` (max-heap by distance) and a
    /// candidate pool (min-heap by distance). Stops when the closest unexpanded
    /// candidate is farther than the worst result.
    fn beam_search(&self, query: &[f32], ef: usize) -> Vec<(u32, f32)> {
        let n = self.num_vectors;
        if n == 0 {
            return Vec::new();
        }

        thread_local! {
            static VISITED: std::cell::RefCell<(Vec<u8>, u8)> =
                const { std::cell::RefCell::new((Vec::new(), 1)) };
        }

        VISITED.with(|cell| {
            let (marks, gen) = &mut *cell.borrow_mut();
            if marks.len() < n {
                marks.resize(n, 0);
            }
            if let Some(next) = gen.checked_add(1) {
                *gen = next;
            } else {
                marks.fill(0);
                *gen = 1;
            }
            let generation = *gen;

            let mut visited_insert = |id: u32| -> bool {
                let idx = id as usize;
                if idx < marks.len() && marks[idx] != generation {
                    marks[idx] = generation;
                    true
                } else { idx >= marks.len() }
            };

            // Min-heap: candidates to expand (closest first)
            let mut candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
            // Max-heap: best results (farthest first, for bounded eviction)
            let mut results: BinaryHeap<(FloatOrd, u32)> = BinaryHeap::new();

            let entry = self.medoid;
            let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
            visited_insert(entry);
            candidates.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
            results.push((FloatOrd(entry_dist), entry));

            while let Some(std::cmp::Reverse((FloatOrd(cand_dist), cand_id))) = candidates.pop() {
                // Stop when closest candidate is worse than worst result
                let worst_dist = results.peek().map_or(f32::INFINITY, |&(FloatOrd(d), _)| d);
                if results.len() >= ef && cand_dist > worst_dist {
                    break;
                }

                let neighbors = &self.neighbors[cand_id as usize];
                for (i, &neighbor) in neighbors.iter().enumerate() {
                    // Prefetch next neighbor's vector
                    if i + 1 < neighbors.len() {
                        let next_id = neighbors[i + 1] as usize;
                        let ptr = self.vectors.as_ptr().wrapping_add(next_id * self.dimension);
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
                        }
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                        }
                    }

                    if visited_insert(neighbor) {
                        let dist =
                            cosine_distance_normalized(query, self.get_vector(neighbor as usize));

                        let worst_dist = results.peek().map_or(f32::INFINITY, |&(FloatOrd(d), _)| d);
                        if results.len() < ef || dist < worst_dist {
                            candidates.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
                            results.push((FloatOrd(dist), neighbor));
                            if results.len() > ef {
                                results.pop(); // evict farthest
                            }
                        }
                    }
                }
            }

            let mut out: Vec<(u32, f32)> = results
                .into_iter()
                .map(|(FloatOrd(d), id)| (id, d))
                .collect();
            out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            out
        })
    }

    /// Beam search with a custom distance function.
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
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }
        let ef = ef.max(k);
        let n = self.num_vectors;
        if n == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        thread_local! {
            static VISITED_SD: std::cell::RefCell<(Vec<u8>, u8)> =
                const { std::cell::RefCell::new((Vec::new(), 1)) };
        }

        VISITED_SD.with(|cell| {
            let (marks, gen) = &mut *cell.borrow_mut();
            if marks.len() < n {
                marks.resize(n, 0);
            }
            if let Some(next) = gen.checked_add(1) {
                *gen = next;
            } else {
                marks.fill(0);
                *gen = 1;
            }
            let generation = *gen;

            let mut visited_insert = |id: u32| -> bool {
                let idx = id as usize;
                if idx < marks.len() && marks[idx] != generation {
                    marks[idx] = generation;
                    true
                } else { idx >= marks.len() }
            };

            let mut candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
            let mut results: BinaryHeap<(FloatOrd, u32)> = BinaryHeap::new();

            let entry = self.medoid;
            let entry_dist = dist_fn(query, entry);
            visited_insert(entry);
            candidates.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
            results.push((FloatOrd(entry_dist), entry));

            while let Some(std::cmp::Reverse((FloatOrd(cand_dist), cand_id))) = candidates.pop() {
                let worst_dist = results.peek().map_or(f32::INFINITY, |&(FloatOrd(d), _)| d);
                if results.len() >= ef && cand_dist > worst_dist {
                    break;
                }

                let neighbors = &self.neighbors[cand_id as usize];
                for (i, &neighbor) in neighbors.iter().enumerate() {
                    // Prefetch next neighbor's vector (no-op for custom dist_fn, but
                    // the neighbor list itself benefits from prefetch)
                    if i + 1 < neighbors.len() {
                        let next_id = neighbors[i + 1] as usize;
                        let ptr = self.vectors.as_ptr().wrapping_add(next_id * self.dimension);
                        #[cfg(target_arch = "aarch64")]
                        unsafe {
                            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
                        }
                        #[cfg(target_arch = "x86_64")]
                        unsafe {
                            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
                        }
                    }

                    if visited_insert(neighbor) {
                        let dist = dist_fn(query, neighbor);

                        let worst_dist = results.peek().map_or(f32::INFINITY, |&(FloatOrd(d), _)| d);
                        if results.len() < ef || dist < worst_dist {
                            candidates.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
                            results.push((FloatOrd(dist), neighbor));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }

            let mut out: Vec<(u32, f32)> = results
                .into_iter()
                .map(|(FloatOrd(d), id)| (self.doc_ids[id as usize], d))
                .collect();
            out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            out.truncate(k);
            Ok(out)
        })
    }

    fn ensure_connectivity(&mut self) {
        let (dim, vecs) = (self.dimension, &self.vectors);
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.medoid, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }
}

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

    #[test]
    fn build_and_search() {
        let dim = 16;
        let n = 40;
        let data = make_vectors(n, dim, 42);

        let mut index = NsgIndex::new(
            dim,
            NsgParams {
                max_degree: 16,
                pool_size: 32,
                knn_degree: 16,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();

        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().any(|(id, _)| *id == 0));
    }

    #[test]
    fn self_search_recall() {
        let dim = 16;
        let n = 30;
        let data = make_vectors(n, dim, 7);

        let mut index = NsgIndex::new(
            dim,
            NsgParams {
                max_degree: 16,
                pool_size: 32,
                knn_degree: 16,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
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
            recall > 0.6,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    #[test]
    fn connectivity() {
        let dim = 8;
        let n = 20;
        let data = make_vectors(n, dim, 123);

        let mut index = NsgIndex::new(
            dim,
            NsgParams {
                max_degree: 8,
                pool_size: 16,
                knn_degree: 8,
                ef_search: 30,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();

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
        assert_eq!(reachable, n);
    }

    #[test]
    fn mrng_prunes_degree() {
        let dim = 16;
        let n = 50;
        let data = make_vectors(n, dim, 99);

        let mut index = NsgIndex::new(
            dim,
            NsgParams {
                max_degree: 32,
                pool_size: 40,
                knn_degree: 32,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();

        // MRNG pruning should produce avg degree < max_degree
        let avg_deg: f64 = index.neighbors.iter().map(|n| n.len() as f64).sum::<f64>() / n as f64;
        assert!(avg_deg < 32.0, "avg degree {avg_deg:.1} should be < 32");
    }

    #[test]
    fn empty_index_errors() {
        let mut index = NsgIndex::new(8, NsgParams::default()).unwrap();
        assert!(index.build().is_err());
    }
}
