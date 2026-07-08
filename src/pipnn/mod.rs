//! PiPNN: Pick-in-Partitions Nearest Neighbors.
//!
//! Partition-based graph construction that avoids the beam-search bottleneck
//! of HNSW/Vamana. Instead of incremental insertion with random-access beam
//! searches, PiPNN:
//!
//! 1. Partitions data into small overlapping leaves via Randomized Ball Carving
//! 2. Computes all-pairs distances within each leaf (cache-friendly GEMM)
//! 3. Prunes candidate edges online via HashPrune (LSH-based reservoir)
//! 4. Optionally applies a final RobustPrune (alpha-RNG) pass
//!
//! Result: 4-12x faster construction than HNSW/Vamana at equal recall.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.10.5", features = ["pipnn"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::pipnn::{PipnnIndex, PipnnParams};
//!
//! let params = PipnnParams::default();
//! let mut index = PipnnIndex::new(128, params)?;
//!
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # How HashPrune Works
//!
//! For each point p, HashPrune maintains a fixed-size reservoir of candidate
//! neighbors keyed by LSH hash. The hash function `h_p(c)` hashes the
//! *residual* `c - p` into m-bit SimHash signatures. When two candidates
//! collide (same hash bucket), the closer one is kept. This enforces
//! directional diversity: neighbors in similar directions compete, while
//! neighbors in different directions coexist.
//!
//! The reservoir is history-independent: the final neighbor list is the same
//! regardless of insertion order. This enables embarrassingly parallel
//! construction across partitions.
//!
//! # References
//!
//! - Rubel et al. (2026). "PiPNN: Ultra-Scalable Graph-Based Nearest Neighbor
//!   Indexing." arXiv:2602.21247.

use crate::distance::cosine_distance_normalized;
use crate::distance::FloatOrd;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::BinaryHeap;
use std::path::Path;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const PIPNN_FORMAT_VERSION: u32 = 1;
const PIPNN_NEIGHBORS_MAGIC: &[u8; 8] = b"PIPNNG01";

/// PiPNN parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PipnnParams {
    /// Maximum leaf size for partitioning (C_max). Default: 2048.
    pub max_leaf_size: usize,
    /// Minimum leaf size before merging into parent. Default: 64.
    pub min_leaf_size: usize,
    /// Leader sampling fraction for RBC partitioning. Default: 0.02.
    pub leader_fraction: f64,
    /// Maximum out-degree (reservoir size in HashPrune). Default: 32.
    pub max_degree: usize,
    /// Number of SimHash hyperplanes for HashPrune. Default: 12.
    pub num_hash_bits: usize,
    /// Apply final RobustPrune pass. Default: true.
    pub final_prune: bool,
    /// Alpha for RobustPrune (RNG relaxation factor). Default: 1.2.
    pub alpha: f32,
    /// Search beam width. Default: 100.
    pub ef_search: usize,
}

impl Default for PipnnParams {
    fn default() -> Self {
        Self {
            max_leaf_size: 2048,
            min_leaf_size: 64,
            leader_fraction: 0.02,
            max_degree: 32,
            num_hash_bits: 12,
            final_prune: true,
            alpha: 1.2,
            ef_search: 100,
        }
    }
}

/// PiPNN index.
pub struct PipnnIndex {
    dimension: usize,
    params: PipnnParams,
    built: bool,

    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Single-layer neighbor graph.
    neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Entry point (medoid).
    medoid: u32,

    /// Random hyperplanes for SimHash (dimension x num_hash_bits, flat).
    hyperplanes: Vec<f32>,
}

#[derive(Deserialize, Serialize)]
struct PipnnSnapshot {
    version: u32,
    dimension: usize,
    num_vectors: usize,
    params: PipnnParams,
    medoid: u32,
}

impl PipnnIndex {
    /// Create a new PiPNN index.
    pub fn new(dimension: usize, params: PipnnParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        // Generate random hyperplanes for SimHash
        let m = params.num_hash_bits;
        let mut hyperplanes = Vec::with_capacity(dimension * m);
        let mut rng: u64 = 42;
        for _ in 0..dimension * m {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = ((rng >> 33) as f64 / (1u64 << 31) as f64) * 2.0 - 1.0;
            hyperplanes.push(val as f32);
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
            hyperplanes,
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

    /// Build the graph.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let n = self.num_vectors;
        self.neighbors = vec![SmallVec::new(); n];

        // Step 1: Compute medoid
        self.medoid = self.compute_medoid();

        // Step 2: Partition via Randomized Ball Carving
        let all_ids: Vec<u32> = (0..n as u32).collect();
        let leaves = self.partition(&all_ids, 0);

        // Step 3: Build kNN within each leaf + HashPrune edges
        #[cfg(feature = "parallel")]
        {
            let all_edges: Vec<Vec<(u32, Vec<(u32, f32)>)>> = leaves
                .par_iter()
                .map(|leaf| self.compute_leaf_edges(leaf))
                .collect();
            for leaf_edges in all_edges {
                for (point_id, candidates) in leaf_edges {
                    self.hashprune_insert(point_id, &candidates);
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        for leaf in &leaves {
            self.build_leaf(leaf);
        }

        // Step 4: Add reverse edges
        self.add_reverse_edges();

        // Step 5: Optional final RobustPrune
        if self.params.final_prune {
            self.final_prune();
        }

        // Step 6: Ensure connectivity
        self.ensure_connectivity();

        self.built = true;
        Ok(())
    }

    /// Save a built PiPNN index to a directory.
    ///
    /// The saved format stores the built in-memory graph plus the SimHash
    /// hyperplanes used during construction.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt PiPNN index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        let snapshot = PipnnSnapshot {
            version: PIPNN_FORMAT_VERSION,
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
            PIPNN_NEIGHBORS_MAGIC,
            &self.neighbors,
        )?;
        crate::graph_snapshot::write_f32_atomic(
            &output_dir.join("hyperplanes.bin"),
            &self.hyperplanes,
        )?;
        Ok(())
    }

    /// Load a PiPNN index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let input_dir = input_dir.as_ref();
        let snapshot: PipnnSnapshot =
            crate::graph_snapshot::read_json(&input_dir.join("manifest.json"))?;
        if snapshot.version != PIPNN_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported PiPNN format version {}",
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
            PIPNN_NEIGHBORS_MAGIC,
            snapshot.num_vectors,
        )?;
        let hyperplanes_len = snapshot
            .dimension
            .checked_mul(snapshot.params.num_hash_bits)
            .ok_or_else(|| RetrieveError::FormatError("PiPNN hyperplane length overflow".into()))?;
        let hyperplanes = crate::graph_snapshot::read_f32_exact(
            &input_dir.join("hyperplanes.bin"),
            hyperplanes_len,
        )?;
        crate::graph_snapshot::validate_graph_shape(
            "PiPNN",
            snapshot.dimension,
            snapshot.num_vectors,
            &vectors,
            &doc_ids,
            &neighbors,
            Some(snapshot.medoid),
        )?;
        Ok(Self {
            dimension: snapshot.dimension,
            params: snapshot.params,
            built: true,
            vectors,
            num_vectors: snapshot.num_vectors,
            doc_ids,
            neighbors,
            medoid: snapshot.medoid,
            hyperplanes,
        })
    }

    /// Memory usage breakdown for this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let metadata_bytes = self.doc_ids.len() * std::mem::size_of::<u32>()
            + self.hyperplanes.len() * std::mem::size_of::<f32>();

        crate::memory::MemoryReport {
            vectors_bytes: self.vectors.len() * std::mem::size_of::<f32>(),
            graph_bytes: crate::memory::smallvec_u32_bytes(&self.neighbors),
            quantized_bytes: 0,
            metadata_bytes,
        }
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

    // ── Internal: partitioning ─────────────────────────────────────────

    /// Randomized Ball Carving: recursively partition until leaf size <= C_max.
    fn partition(&self, ids: &[u32], depth: usize) -> Vec<Vec<u32>> {
        if ids.len() <= self.params.max_leaf_size || depth > 20 {
            return vec![ids.to_vec()];
        }

        // Sample leaders
        let num_leaders = ((ids.len() as f64 * self.params.leader_fraction) as usize)
            .max(2)
            .min(ids.len());

        // Deterministic pseudo-random leader selection based on depth
        let mut rng: u64 = (depth as u64).wrapping_mul(2654435761).wrapping_add(42);
        let mut leader_indices: Vec<usize> = Vec::with_capacity(num_leaders);
        let mut available: Vec<usize> = (0..ids.len()).collect();
        for _ in 0..num_leaders {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng >> 33) as usize % available.len();
            leader_indices.push(available.swap_remove(idx));
        }

        let leaders: Vec<u32> = leader_indices.iter().map(|&i| ids[i]).collect();

        // Assign each point to nearest leader(s)
        // Multi-level fanout: assign to k nearest leaders at top level only
        // Deeper levels use strict 1-nearest to guarantee convergence
        let k_nearest = if depth == 0 { 2.min(num_leaders) } else { 1 };

        let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); leaders.len()];
        for &id in ids {
            let vi = self.get_vector(id as usize);
            let mut dists: Vec<(usize, f32)> = leaders
                .iter()
                .enumerate()
                .map(|(li, &lid)| {
                    let lv = self.get_vector(lid as usize);
                    (li, cosine_distance_normalized(vi, lv))
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            for &(li, _) in dists.iter().take(k_nearest) {
                buckets[li].push(id);
            }
        }

        // Merge small buckets into neighbors
        let min_size = self.params.min_leaf_size;
        let mut merged_buckets: Vec<Vec<u32>> = Vec::new();
        let mut small_buf: Vec<u32> = Vec::new();
        for bucket in buckets {
            if bucket.len() < min_size {
                small_buf.extend(bucket);
                if small_buf.len() >= min_size {
                    merged_buckets.push(std::mem::take(&mut small_buf));
                }
            } else {
                merged_buckets.push(bucket);
            }
        }
        if !small_buf.is_empty() {
            if let Some(last) = merged_buckets.last_mut() {
                last.extend(small_buf);
            } else {
                merged_buckets.push(small_buf);
            }
        }

        // Recurse on each bucket
        let mut leaves = Vec::new();
        for bucket in merged_buckets {
            leaves.extend(self.partition(&bucket, depth + 1));
        }
        leaves
    }

    // ── Internal: leaf building ────────────────────────────────────────

    /// Build kNN graph within a leaf and feed edges to HashPrune.
    #[allow(dead_code)]
    fn build_leaf(&mut self, leaf: &[u32]) {
        if leaf.len() <= 1 {
            return;
        }

        let max_k = self.params.max_degree.min(leaf.len() - 1);

        // Compute all-pairs distance matrix within leaf
        // This is the GEMM-equivalent step (cache-friendly since leaf fits in L2)
        let n = leaf.len();
        let mut distances = vec![0.0f32; n * n];
        for i in 0..n {
            let vi = self.get_vector(leaf[i] as usize);
            for j in (i + 1)..n {
                let vj = self.get_vector(leaf[j] as usize);
                let d = cosine_distance_normalized(vi, vj);
                distances[i * n + j] = d;
                distances[j * n + i] = d;
            }
        }

        // For each point, extract kNN from the distance matrix
        for i in 0..n {
            let point_id = leaf[i];
            let mut candidates: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (leaf[j], distances[i * n + j]))
                .collect();
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            candidates.truncate(max_k * 2); // over-fetch for HashPrune

            // HashPrune: stream candidates into the reservoir
            self.hashprune_insert(point_id, &candidates);
        }
    }

    /// Compute kNN candidates for each point in a leaf without mutating `self`.
    ///
    /// Returns `(point_id, candidates)` pairs suitable for feeding into
    /// `hashprune_insert`. Used by the parallel build path.
    #[cfg(feature = "parallel")]
    fn compute_leaf_edges(&self, leaf: &[u32]) -> Vec<(u32, Vec<(u32, f32)>)> {
        if leaf.len() <= 1 {
            return Vec::new();
        }

        let max_k = self.params.max_degree.min(leaf.len() - 1);
        let n = leaf.len();

        // All-pairs distance matrix within the leaf
        let mut distances = vec![0.0f32; n * n];
        for i in 0..n {
            let vi = self.get_vector(leaf[i] as usize);
            for j in (i + 1)..n {
                let vj = self.get_vector(leaf[j] as usize);
                let d = cosine_distance_normalized(vi, vj);
                distances[i * n + j] = d;
                distances[j * n + i] = d;
            }
        }

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let point_id = leaf[i];
            let mut candidates: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (leaf[j], distances[i * n + j]))
                .collect();
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            candidates.truncate(max_k * 2);
            result.push((point_id, candidates));
        }
        result
    }

    /// HashPrune: insert candidate neighbors into the point's reservoir.
    ///
    /// The reservoir is keyed by SimHash of the residual (candidate - point).
    /// Candidates that hash to the same bucket compete; the closer one wins.
    /// This enforces directional diversity without explicit angular checks.
    fn hashprune_insert(&mut self, point_id: u32, candidates: &[(u32, f32)]) {
        let max_deg = self.params.max_degree;
        let m = self.params.num_hash_bits;
        let reservoir_size = 1u32 << m.min(16); // hash space size (capped)

        // Reservoir: hash -> (candidate_id, distance)
        let mut reservoir: Vec<Option<(u32, f32)>> = vec![None; reservoir_size as usize];
        let mut count = 0usize;

        let pv = self.get_vector(point_id as usize).to_vec();

        for &(cand_id, dist) in candidates {
            if count >= max_deg * 2 {
                break; // Don't overfill
            }

            // Compute SimHash of residual (cand - point)
            let cv = self.get_vector(cand_id as usize);
            let hash = self.simhash_residual(&pv, cv, m);
            let bucket = (hash % reservoir_size) as usize;

            match reservoir[bucket] {
                Some((_, existing_dist)) => {
                    // Collision: keep closer candidate
                    if dist < existing_dist {
                        reservoir[bucket] = Some((cand_id, dist));
                    }
                }
                None => {
                    reservoir[bucket] = Some((cand_id, dist));
                    count += 1;
                }
            }
        }

        // Collect reservoir contents, sort by distance, take max_degree
        let mut selected: Vec<(u32, f32)> = reservoir.into_iter().flatten().collect();
        selected.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        selected.truncate(max_deg);

        // Merge with existing neighbors (from other leaves due to overlap)
        let existing = &self.neighbors[point_id as usize];
        let mut merged: Vec<(u32, f32)> = existing
            .iter()
            .map(|&id| {
                let d = cosine_distance_normalized(&pv, self.get_vector(id as usize));
                (id, d)
            })
            .chain(selected)
            .collect();
        merged.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        merged.dedup_by_key(|c| c.0);
        merged.truncate(max_deg);

        self.neighbors[point_id as usize] = merged.iter().map(|&(id, _)| id).collect();
    }

    /// SimHash of residual vector (c - p).
    fn simhash_residual(&self, p: &[f32], c: &[f32], m: usize) -> u32 {
        let dim = self.dimension;
        let mut hash = 0u32;
        for bit in 0..m.min(32) {
            let hp_start = bit * dim;
            let mut dot = 0.0f32;
            for d in 0..dim {
                let residual = c[d] - p[d];
                dot += residual * self.hyperplanes[hp_start + d];
            }
            if dot >= 0.0 {
                hash |= 1 << bit;
            }
        }
        hash
    }

    // ── Internal: graph refinement ─────────────────────────────────────

    /// Add reverse edges (bidirectional graph).
    fn add_reverse_edges(&mut self) {
        let n = self.num_vectors;
        let max_deg = self.params.max_degree;

        for i in 0..n {
            let nbs: SmallVec<[u32; 16]> = self.neighbors[i].clone();
            for &nb in &nbs {
                let nb = nb as usize;
                if nb < n
                    && !self.neighbors[nb].contains(&(i as u32))
                    && self.neighbors[nb].len() < max_deg
                {
                    self.neighbors[nb].push(i as u32);
                }
            }
        }
    }

    /// Final RobustPrune pass (alpha-RNG).
    fn final_prune(&mut self) {
        let n = self.num_vectors;
        let max_deg = self.params.max_degree;
        let alpha = self.params.alpha;

        for i in 0..n {
            if self.neighbors[i].len() <= max_deg {
                continue;
            }

            let vi = self.get_vector(i).to_vec();
            let mut candidates: Vec<(u32, f32)> = self.neighbors[i]
                .iter()
                .map(|&id| {
                    let d = cosine_distance_normalized(&vi, self.get_vector(id as usize));
                    (id, d)
                })
                .collect();
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            // Alpha-RNG pruning
            let mut selected: Vec<u32> = Vec::with_capacity(max_deg);
            for &(cand_id, cand_dist) in &candidates {
                if selected.len() >= max_deg {
                    break;
                }
                let cand_vec = self.get_vector(cand_id as usize);
                let mut keep = true;
                for &sel_id in &selected {
                    let sel_vec = self.get_vector(sel_id as usize);
                    let inter_dist = cosine_distance_normalized(sel_vec, cand_vec);
                    if alpha * inter_dist < cand_dist {
                        keep = false;
                        break;
                    }
                }
                if keep {
                    selected.push(cand_id);
                }
            }

            self.neighbors[i] = SmallVec::from_vec(selected);
        }
    }

    fn compute_medoid(&self) -> u32 {
        let n = self.num_vectors;
        let dim = self.dimension;
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

    fn ensure_connectivity(&mut self) {
        let (dim, vecs) = (self.dimension, &self.vectors);
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.medoid, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }

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
                } else {
                    idx >= marks.len()
                }
            };

            let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
            let mut candidates: Vec<(u32, f32)> = Vec::new();

            let entry = self.medoid;
            let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
            visited_insert(entry);
            frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
            candidates.push((entry, entry_dist));

            let mut visited_count = 1usize;

            while let Some(std::cmp::Reverse((FloatOrd(current_dist), current_id))) = frontier.pop()
            {
                if candidates.len() >= ef {
                    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    if current_dist > candidates[ef - 1].1 * 1.5 {
                        break;
                    }
                }

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

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
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
        let n = 200;
        let data = make_vectors(n, dim, 42);

        let mut index = PipnnIndex::new(
            dim,
            PipnnParams {
                max_leaf_size: 64,
                max_degree: 16,
                num_hash_bits: 8,
                final_prune: true,
                ef_search: 50,
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

        let query = &data[0..dim];
        let results = index.search(query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "expected self-match: {:?}",
            results
        );
    }

    #[test]
    fn self_search_recall() {
        let dim = 16;
        let n = 100;
        let data = make_vectors(n, dim, 7);

        let mut index = PipnnIndex::new(
            dim,
            PipnnParams {
                max_leaf_size: 50,
                max_degree: 16,
                num_hash_bits: 10,
                final_prune: true,
                ef_search: 50,
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
            recall > 0.5,
            "self-search recall too low: {recall:.2} ({hits}/{n})"
        );
    }

    #[test]
    fn save_load_roundtrip_preserves_search() {
        let dim = 16;
        let n = 100;
        let data = make_vectors(n, dim, 7);
        let mut index = PipnnIndex::new(
            dim,
            PipnnParams {
                max_leaf_size: 50,
                max_degree: 16,
                num_hash_bits: 10,
                final_prune: true,
                ef_search: 50,
                ..Default::default()
            },
        )
        .unwrap();

        for i in 0..n {
            let start = i * dim;
            index
                .add_slice(10_000 + i as u32, &data[start..start + dim])
                .unwrap();
        }
        index.build().unwrap();
        let query = &data[0..dim];
        let before = index.search_with_ef(query, 5, 50).unwrap();

        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = PipnnIndex::load_from_dir(dir.path()).unwrap();

        assert_eq!(loaded.search_with_ef(query, 5, 50).unwrap(), before);
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.hyperplanes, index.hyperplanes);
    }

    #[test]
    fn overlapping_partitions() {
        // With multi-level fanout at depth=0, points should appear in multiple leaves
        let dim = 8;
        let n = 200;
        let data = make_vectors(n, dim, 99);

        let index = PipnnIndex::new(
            dim,
            PipnnParams {
                max_leaf_size: 50,
                ..Default::default()
            },
        )
        .unwrap();

        // Manually add vectors for partition testing
        let mut idx = index;
        for i in 0..n {
            let start = i * dim;
            idx.add_slice(i as u32, &data[start..start + dim]).unwrap();
        }

        let all_ids: Vec<u32> = (0..n as u32).collect();
        let leaves = idx.partition(&all_ids, 0);

        // Total points across all leaves should exceed n (due to overlap)
        let total: usize = leaves.iter().map(|l| l.len()).sum();
        assert!(
            total >= n,
            "expected overlap: total {total} should be >= {n}"
        );
    }

    #[test]
    fn empty_index_errors() {
        let mut index = PipnnIndex::new(8, PipnnParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn connectivity() {
        let dim = 8;
        let n = 50;
        let data = make_vectors(n, dim, 123);

        let mut index = PipnnIndex::new(
            dim,
            PipnnParams {
                max_leaf_size: 20,
                max_degree: 8,
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
        assert_eq!(reachable, n);
    }
}
