//! delta-EMG: Error-Bounded Monotonic Graph for ANN search.
//!
//! A single-layer proximity graph where every greedy walk is guaranteed to
//! terminate within a `(1/delta)`-approximate nearest neighbor. Unlike HNSW
//! (which provides no per-query distance guarantee) or Vamana (which uses a
//! scalar alpha relaxation), delta-EMG encodes the approximation bound
//! directly into the graph structure via occlusion-based edge pruning.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["emg"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::emg::{EmgIndex, EmgParams};
//!
//! let params = EmgParams {
//!     max_degree: 32,
//!     candidate_size: 64,
//!     ..Default::default()
//! };
//! let mut index = EmgIndex::new(128, params)?;
//!
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # Approximation Guarantee
//!
//! For any query q and search result r returned by greedy search on a delta-EMG:
//!
//! $$d(q, r) \leq \frac{1}{\delta} \cdot d(q, v_1)$$
//!
//! where $v_1$ is the true nearest neighbor. Smaller delta = weaker guarantee
//! but sparser graph (faster search). The practical variant uses adaptive delta
//! that varies per-edge based on neighbor distance rank.
//!
//! # How It Works: Occlusion Pruning
//!
//! When building node u's neighbor list from sorted candidates, candidate v is
//! added only if no already-selected neighbor w falls in the occlusion region:
//!
//! $$\text{Occ}_\delta(u, v) = \{x : d(x,u) < d(u,v) \wedge d^2(x,v) + 2\delta \cdot d(u,v) \cdot d(x,u) < d^2(u,v)\}$$
//!
//! As delta approaches 0, the occlusion region expands toward the MRNG lune
//! (strictest pruning, fewest edges). As delta approaches 1, it contracts
//! (more edges, tighter guarantee).
//!
//! The adaptive variant sets `delta_t(u,v) = 1 - d(u,v) / d(u, v_t)` where
//! `v_t` is the t-th closest candidate. Long edges are pruned aggressively,
//! short edges are retained. Expected out-degree: O(ln n).
//!
//! # References
//!
//! - Yin et al. (2025). "delta-EMG: Error-Bounded Monotonic Graph for
//!   Approximate Nearest Neighbor Search." arXiv:2511.16921.

use crate::distance::cosine_distance_normalized;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::BinaryHeap;
use std::path::Path;

const EMG_FORMAT_VERSION: u32 = 1;
const EMG_NEIGHBORS_MAGIC: &[u8; 8] = b"EMGGRPH1";

/// delta-EMG parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EmgParams {
    /// Maximum out-degree per node.
    pub max_degree: usize,
    /// Candidate set size during construction (L in the paper).
    pub candidate_size: usize,
    /// Scale parameter for adaptive delta (t in the paper, t <= candidate_size).
    /// The t-th closest candidate determines the local scale.
    pub scale_t: usize,
    /// Construction iterations. More iterations = better graph quality.
    pub iterations: usize,
    /// Search termination factor. Larger alpha = more exploration.
    /// Search stops when `C[l] >= alpha * C[k]`.
    pub alpha: f32,
    /// Default ef_search.
    pub ef_search: usize,
}

impl Default for EmgParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            candidate_size: 64,
            scale_t: 32,
            iterations: 2,
            alpha: 1.5,
            ef_search: 100,
        }
    }
}

/// delta-EMG index.
///
/// **Cosine distance only.** Input vectors are L2-normalized in
/// [`Self::add_slice`]; pre-normalization is optional. The
/// [`crate::distance::DistanceMetric`] enum is not honored.
pub struct EmgIndex {
    dimension: usize,
    params: EmgParams,
    built: bool,

    /// Flat vector storage.
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Single-layer neighbor lists.
    neighbors: Vec<SmallVec<[u32; 32]>>,

    /// Medoid (closest to centroid), used as search entry point.
    medoid: u32,

    /// Scalar-quantized vectors: n * dimension bytes (populated during build).
    quantized_vectors: Vec<u8>,
    /// Per-dimension min values for dequantization.
    quant_mins: Vec<f32>,
    /// Per-dimension scale factors for dequantization.
    quant_scales: Vec<f32>,
}

#[derive(Deserialize, Serialize)]
struct EmgSnapshot {
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: EmgParams,
    medoid: u32,
}

impl EmgIndex {
    /// Create a new delta-EMG index.
    pub fn new(dimension: usize, params: EmgParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        let scale_t = params.scale_t.min(params.candidate_size);
        Ok(Self {
            dimension,
            params: EmgParams { scale_t, ..params },
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            neighbors: Vec::new(),
            medoid: 0,
            quantized_vectors: Vec::new(),
            quant_mins: Vec::new(),
            quant_scales: Vec::new(),
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
        // L2-normalize
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

    /// Build the graph.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Compute medoid
        self.medoid = self.compute_medoid();

        // Initialize with approximate kNN graph (random + one pass of NN-descent-like refinement)
        self.initialize_random_graph();

        // Iterative refinement with occlusion pruning
        for _ in 0..self.params.iterations {
            self.refine_pass()?;
        }

        // Quantize all vectors for fast approximate search
        self.quantize_vectors();

        self.built = true;
        Ok(())
    }

    /// Save a built delta-EMG index to a directory.
    ///
    /// The saved format stores the built in-memory graph. Loading restores the
    /// graph directly and rebuilds deterministic scalar quantization payloads.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt EMG index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let snapshot = EmgSnapshot {
            version: EMG_FORMAT_VERSION,
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
            EMG_NEIGHBORS_MAGIC,
            &self.neighbors,
        )?;
        Ok(())
    }

    /// Load a delta-EMG index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let snapshot: EmgSnapshot =
            crate::graph_snapshot::read_json(&input_dir.join("manifest.json"))?;
        if snapshot.version != EMG_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported EMG format version {}",
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
            EMG_NEIGHBORS_MAGIC,
            snapshot.num_vectors,
        )?;
        crate::graph_snapshot::validate_graph_shape(
            "EMG",
            snapshot.dimension,
            snapshot.num_vectors,
            &vectors,
            &doc_ids,
            &neighbors,
            Some(snapshot.medoid),
        )?;

        let mut index = Self {
            dimension: snapshot.dimension,
            params: snapshot.params,
            built: true,
            vectors,
            num_vectors: snapshot.num_vectors,
            doc_ids,
            neighbors,
            medoid: snapshot.medoid,
            quantized_vectors: Vec::new(),
            quant_mins: Vec::new(),
            quant_scales: Vec::new(),
        };
        index.quantize_vectors();
        Ok(index)
    }

    /// Memory usage breakdown for this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let quantized_bytes = self.quantized_vectors.len()
            + (self.quant_mins.len() + self.quant_scales.len()) * std::mem::size_of::<f32>();

        crate::memory::MemoryReport {
            vectors_bytes: self.vectors.len() * std::mem::size_of::<f32>(),
            graph_bytes: crate::memory::smallvec_u32_bytes(&self.neighbors),
            quantized_bytes,
            metadata_bytes: self.doc_ids.len() * std::mem::size_of::<u32>(),
        }
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Search with custom ef.
    pub fn search_with_ef(
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

        let results = self.greedy_search(&query_normalized, ef.max(k));

        Ok(results
            .into_iter()
            .take(k)
            .map(|(id, dist)| (self.doc_ids[id as usize], dist))
            .collect())
    }

    /// Search using scalar-quantized distance for beam expansion, then rerank
    /// the top `k * rerank_factor` candidates with exact cosine distance.
    ///
    /// Faster than `search_with_ef` at the cost of a small accuracy reduction.
    /// `rerank_factor` controls the accuracy/speed tradeoff: 1 is fastest
    /// (no reranking headroom), 4–8 recovers most of the accuracy loss.
    pub fn search_quantized(
        &self,
        query: &[f32],
        k: usize,
        rerank_factor: usize,
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

        // Quantize query using stored per-dimension min/scale
        let dim = self.dimension;
        let mut query_quantized = vec![0u8; dim];
        for d in 0..dim {
            let scale = self.quant_scales[d];
            if scale > 1e-10 {
                let q = (query_normalized[d] - self.quant_mins[d]) * scale;
                query_quantized[d] = q.clamp(0.0, 255.0) as u8;
            }
        }

        // Beam search with quantized distances
        let ef = (k * rerank_factor).max(k).max(self.params.ef_search);
        let candidates = self.greedy_search_quantized(&query_quantized, ef);

        // Rerank top k*rerank_factor with exact distance
        let rerank_pool = k * rerank_factor.max(1);
        let mut reranked: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(rerank_pool)
            .map(|(internal_id, _approx)| {
                let exact = cosine_distance_normalized(
                    &query_normalized,
                    self.get_vector(internal_id as usize),
                );
                (internal_id, exact)
            })
            .collect();
        reranked.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        Ok(reranked
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

    // ── Internal methods ───────────────────────────────────────────────

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Compute per-dimension min/max and quantize all stored vectors to u8.
    fn quantize_vectors(&mut self) {
        let dim = self.dimension;
        let n = self.num_vectors;

        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];

        for i in 0..n {
            let v = self.get_vector(i);
            for d in 0..dim {
                if v[d] < mins[d] {
                    mins[d] = v[d];
                }
                if v[d] > maxs[d] {
                    maxs[d] = v[d];
                }
            }
        }

        let mut scales = vec![0.0f32; dim];
        for d in 0..dim {
            let range = maxs[d] - mins[d];
            if range > 1e-10 {
                scales[d] = 255.0 / range;
            }
        }

        let mut qvecs = vec![0u8; n * dim];
        for i in 0..n {
            let v = self.get_vector(i);
            for d in 0..dim {
                let q = if scales[d] > 1e-10 {
                    ((v[d] - mins[d]) * scales[d]).clamp(0.0, 255.0) as u8
                } else {
                    0u8
                };
                qvecs[i * dim + d] = q;
            }
        }

        self.quant_mins = mins;
        self.quant_scales = scales;
        self.quantized_vectors = qvecs;
    }

    /// L2 distance on quantized (u8) vectors.
    ///
    /// Used as a fast approximation during beam search. Integer arithmetic
    /// avoids f32 multiply-accumulate for every neighbor expansion.
    #[inline]
    fn quantized_distance(&self, qa: &[u8], qb: &[u8]) -> f32 {
        let mut sum: u32 = 0;
        for (&a, &b) in qa.iter().zip(qb.iter()) {
            let diff = a as i32 - b as i32;
            sum += (diff * diff) as u32;
        }
        sum as f32
    }

    /// Greedy search using quantized distances for fast expansion.
    fn greedy_search_quantized(&self, query_quantized: &[u8], ef: usize) -> Vec<(u32, f32)> {
        let n = self.num_vectors;
        if n == 0 {
            return Vec::new();
        }
        let dim = self.dimension;

        thread_local! {
            static VISITED_Q: std::cell::RefCell<(Vec<u8>, u8)> =
                const { std::cell::RefCell::new((Vec::new(), 1)) };
        }

        VISITED_Q.with(|cell| {
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
                } else {
                    idx >= marks.len()
                }
            };

            let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
            let mut candidates: Vec<(u32, f32)> = Vec::new();

            let entry = self.medoid;
            let entry_q = &self.quantized_vectors[entry as usize * dim..entry as usize * dim + dim];
            let entry_dist = self.quantized_distance(query_quantized, entry_q);
            visited_insert(entry);
            frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
            candidates.push((entry, entry_dist));

            let mut visited_count = 1usize;

            while let Some(std::cmp::Reverse((FloatOrd(current_dist), current_id))) = frontier.pop()
            {
                if candidates.len() >= ef {
                    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    if current_dist > candidates[ef - 1].1 * self.params.alpha {
                        break;
                    }
                }

                let neighbors = &self.neighbors[current_id as usize];
                for (i, &neighbor) in neighbors.iter().enumerate() {
                    // Prefetch next neighbor's quantized vector
                    if i + 1 < neighbors.len() {
                        let next_id = neighbors[i + 1] as usize;
                        let ptr = self.quantized_vectors.as_ptr().wrapping_add(next_id * dim);
                        crate::prefetch::prefetch_read_data(ptr);
                    }

                    if visited_insert(neighbor) {
                        visited_count += 1;
                        let nidx = neighbor as usize;
                        let nq = &self.quantized_vectors[nidx * dim..nidx * dim + dim];
                        let dist = self.quantized_distance(query_quantized, nq);
                        candidates.push((neighbor, dist));
                        frontier.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
                    }
                }

                if visited_count > ef * 10 {
                    break;
                }
            }

            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            candidates.dedup_by_key(|c| c.0);
            candidates
        })
    }

    fn compute_medoid(&self) -> u32 {
        let dim = self.dimension;
        let n = self.num_vectors;

        // Compute centroid
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

        // Find closest to centroid
        let mut best_id = 0u32;
        let mut best_dist = f32::INFINITY;
        for i in 0..n {
            let v = self.get_vector(i);
            let dist = cosine_distance_normalized(&centroid, v);
            if dist < best_dist {
                best_dist = dist;
                best_id = i as u32;
            }
        }
        best_id
    }

    fn initialize_random_graph(&mut self) {
        let n = self.num_vectors;
        let m = self.params.max_degree.min(n - 1);
        self.neighbors = vec![SmallVec::new(); n];

        // Simple initialization: connect each node to its nearest ~m neighbors
        // by scanning all vectors. For large n this should use NN-descent, but
        // the iterative refinement will fix the graph quality.
        let scan_limit = (m * 4).min(n);

        for i in 0..n {
            let vi = self.get_vector(i);
            let mut dists: Vec<(u32, f32)> = (0..scan_limit.min(n))
                .filter(|&j| j != i)
                .map(|j| {
                    let vj = self.get_vector(j);
                    (j as u32, cosine_distance_normalized(vi, vj))
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(m);
            self.neighbors[i] = dists.iter().map(|(id, _)| *id).collect();
        }
    }

    /// One refinement pass: for each node, search for candidates and apply
    /// occlusion-based neighbor selection.
    fn refine_pass(&mut self) -> Result<(), RetrieveError> {
        let n = self.num_vectors;

        for current_id in 0..n {
            let current_vec = self.get_vector(current_id).to_vec();

            // Greedy search to find candidates
            let candidates = self.greedy_search(&current_vec, self.params.candidate_size);

            // Select neighbors via occlusion pruning
            let selected =
                self.select_neighbors_occlusion(current_id as u32, &current_vec, &candidates);

            self.neighbors[current_id] = selected.iter().map(|&(id, _)| id).collect();

            // Add reverse edges with degree cap
            let max_deg = self.params.max_degree;
            for &(neighbor_id, _) in &selected {
                let nid = neighbor_id as usize;
                if !self.neighbors[nid].contains(&(current_id as u32)) {
                    if self.neighbors[nid].len() < max_deg {
                        self.neighbors[nid].push(current_id as u32);
                    } else {
                        // Prune reverse neighbor list with occlusion
                        let nid_vec = self.get_vector(nid).to_vec();
                        let rev_candidates: Vec<(u32, f32)> = self.neighbors[nid]
                            .iter()
                            .chain(std::iter::once(&(current_id as u32)))
                            .map(|&id| {
                                let v = self.get_vector(id as usize);
                                (id, cosine_distance_normalized(&nid_vec, v))
                            })
                            .collect();
                        let pruned =
                            self.select_neighbors_occlusion(neighbor_id, &nid_vec, &rev_candidates);
                        self.neighbors[nid] = pruned.iter().map(|&(id, _)| id).collect();
                    }
                }
            }
        }

        // Ensure connectivity: connect unreachable nodes to medoid
        self.ensure_connectivity();

        Ok(())
    }

    /// Select neighbors using occlusion-based pruning with adaptive delta.
    ///
    /// For each candidate v (sorted by distance from u), v is added only if
    /// no already-selected neighbor w falls in the occlusion region
    /// Occ_delta(u, v).
    fn select_neighbors_occlusion(
        &self,
        node_id: u32,
        node_vec: &[f32],
        candidates: &[(u32, f32)],
    ) -> Vec<(u32, f32)> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let mut sorted: Vec<(u32, f32)> = candidates
            .iter()
            .filter(|(id, _)| *id != node_id)
            .copied()
            .collect();
        sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        let max_deg = self.params.max_degree;
        let scale_t = self.params.scale_t.min(sorted.len());

        // v_t = distance to the t-th closest candidate (for adaptive delta)
        let d_vt = if scale_t > 0 && scale_t <= sorted.len() {
            sorted[scale_t - 1].1
        } else {
            sorted.last().map(|(_, d)| *d).unwrap_or(1.0)
        };

        let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_deg);

        for &(candidate_id, d_uv) in &sorted {
            if selected.len() >= max_deg {
                break;
            }

            // Adaptive delta: closer candidates get larger delta (less pruning)
            let delta = if d_vt > 1e-10 {
                (1.0 - d_uv / d_vt).max(0.01) // floor at 0.01 to avoid zero
            } else {
                0.5
            };

            // Check if any selected neighbor occludes this candidate
            let mut occluded = false;
            for &(selected_id, _) in &selected {
                let w_vec = self.get_vector(selected_id as usize);
                let d_wu = cosine_distance_normalized(w_vec, node_vec);
                let d_wv =
                    cosine_distance_normalized(w_vec, self.get_vector(candidate_id as usize));

                // Occlusion condition:
                // 1. d(w, u) < d(u, v)
                // 2. d(w, v)^2 + 2*delta*d(u,v)*d(w,u) < d(u,v)^2
                if d_wu < d_uv {
                    let lhs = d_wv * d_wv + 2.0 * delta * d_uv * d_wu;
                    let rhs = d_uv * d_uv;
                    if lhs < rhs {
                        occluded = true;
                        break;
                    }
                }
            }

            if !occluded {
                selected.push((candidate_id, d_uv));
            }
        }

        selected
    }

    /// Greedy search from medoid, returning sorted (internal_id, distance) pairs.
    fn greedy_search(&self, query: &[f32], ef: usize) -> Vec<(u32, f32)> {
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
                } else {
                    idx >= marks.len()
                }
            };

            // Min-heap for frontier
            let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();

            // Result candidates
            let mut candidates: Vec<(u32, f32)> = Vec::new();

            let entry = self.medoid;
            let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
            visited_insert(entry);
            frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
            candidates.push((entry, entry_dist));

            let mut visited_count = 1usize;

            while let Some(std::cmp::Reverse((FloatOrd(current_dist), current_id))) = frontier.pop()
            {
                // Early termination: if current is worse than ef-th best candidate
                if candidates.len() >= ef {
                    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    if current_dist > candidates[ef - 1].1 * self.params.alpha {
                        break;
                    }
                }

                // Expand neighbors
                let neighbors = &self.neighbors[current_id as usize];
                for (i, &neighbor) in neighbors.iter().enumerate() {
                    // Prefetch next neighbor's vector
                    if i + 1 < neighbors.len() {
                        let next_id = neighbors[i + 1] as usize;
                        let ptr = self.vectors.as_ptr().wrapping_add(next_id * self.dimension);
                        crate::prefetch::prefetch_read_data(ptr);
                    }

                    if visited_insert(neighbor) {
                        visited_count += 1;
                        let dist =
                            cosine_distance_normalized(query, self.get_vector(neighbor as usize));
                        candidates.push((neighbor, dist));
                        frontier.push(std::cmp::Reverse((FloatOrd(dist), neighbor)));
                    }
                }

                if visited_count > ef * 10 {
                    break;
                }
            }

            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            candidates.dedup_by_key(|c| c.0);
            candidates
        })
    }

    fn ensure_connectivity(&mut self) {
        let (dim, vecs) = (self.dimension, &self.vectors);
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.medoid, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }
}

use crate::distance::FloatOrd;

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
    fn build_and_search_basic() {
        let dim = 16;
        let n = 100;
        let data = make_vectors(n, dim, 42);

        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 16,
                candidate_size: 32,
                scale_t: 16,
                iterations: 2,
                alpha: 1.5,
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
        assert!(results.len() <= 5);
        // Self-match should be near top
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "expected doc_id 0 in results: {:?}",
            results
        );
    }

    #[test]
    fn self_search_recall() {
        let dim = 16;
        let n = 80;
        let data = make_vectors(n, dim, 7);

        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 16,
                candidate_size: 40,
                scale_t: 20,
                iterations: 2,
                alpha: 1.5,
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
    fn save_load_roundtrip_preserves_exact_and_quantized_search() {
        let dim = 16;
        let n = 80;
        let data = make_vectors(n, dim, 7);
        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 16,
                candidate_size: 40,
                scale_t: 20,
                iterations: 2,
                alpha: 1.5,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(20_000 + i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();
        let query = &data[0..dim];
        let exact_before = index.search_with_ef(query, 5, 50).unwrap();
        let quantized_before = index.search_quantized(query, 5, 4).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = EmgIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search_with_ef(query, 5, 50).unwrap(), exact_before);
        assert_eq!(
            loaded.search_quantized(query, 5, 4).unwrap(),
            quantized_before
        );
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.quantized_vectors, index.quantized_vectors);
        assert_eq!(loaded.quant_mins, index.quant_mins);
        assert_eq!(loaded.quant_scales, index.quant_scales);
    }

    #[test]
    fn occlusion_pruning_reduces_degree() {
        // Occlusion pruning should produce sparser graphs than raw kNN
        let dim = 16;
        let n = 50;
        let data = make_vectors(n, dim, 99);

        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 32, // high cap
                candidate_size: 40,
                scale_t: 20,
                iterations: 2,
                ..Default::default()
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

        // Average degree should be well below max_degree due to occlusion pruning
        let avg_degree: f64 =
            index.neighbors.iter().map(|n| n.len() as f64).sum::<f64>() / n as f64;

        assert!(
            avg_degree < 32.0,
            "expected avg degree < max_degree (32) due to pruning, got {avg_degree:.1}"
        );
    }

    #[test]
    fn empty_index_errors() {
        let mut index = EmgIndex::new(8, EmgParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut index = EmgIndex::new(8, EmgParams::default()).unwrap();
        assert!(index.add(0, vec![1.0; 16]).is_err());
    }

    #[test]
    fn connectivity_maintained() {
        // After build, all nodes should be reachable from medoid
        let dim = 8;
        let n = 30;
        let data = make_vectors(n, dim, 123);

        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 8,
                candidate_size: 16,
                scale_t: 8,
                iterations: 2,
                ..Default::default()
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

        // BFS from medoid should reach all nodes
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
        assert_eq!(
            reachable, n,
            "expected all {n} nodes reachable, got {reachable}"
        );
    }

    #[test]
    fn quantized_search_basic() {
        let dim = 16;
        let n = 100;
        let data = make_vectors(n, dim, 77);

        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 16,
                candidate_size: 32,
                scale_t: 16,
                iterations: 2,
                alpha: 1.5,
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
        let results = index.search_quantized(query, 5, 4).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Self-match should appear (reranking with exact distance recovers it)
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "expected doc_id 0 in quantized results: {:?}",
            results
        );
    }

    #[test]
    fn quantized_vs_exact_recall() {
        let dim = 16;
        let n = 100;
        let k = 10;
        let data = make_vectors(n, dim, 55);

        let mut index = EmgIndex::new(
            dim,
            EmgParams {
                max_degree: 16,
                candidate_size: 40,
                scale_t: 20,
                iterations: 2,
                alpha: 1.5,
                ef_search: 60,
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

        let mut total_hits = 0usize;
        let num_queries = 20usize;
        for q in 0..num_queries {
            let query = &data[q * dim..(q + 1) * dim];
            let exact: std::collections::HashSet<u32> = index
                .search_with_ef(query, k, 200)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let approx = index.search_quantized(query, k, 4).unwrap();
            let hits = approx.iter().filter(|(id, _)| exact.contains(id)).count();
            total_hits += hits;
        }

        let recall = total_hits as f64 / (num_queries * k) as f64;
        assert!(
            recall > 0.3,
            "quantized recall@{k} too low: {recall:.2} ({total_hits}/{} hits)",
            num_queries * k
        );
    }
}
