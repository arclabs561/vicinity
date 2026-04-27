//! Mutable proximity graph with incremental insert, delete, and compact.
//! Inspired by FreshDiskANN (Singh et al., 2021) and IP-DiskANN (Xu et al., 2025).
//!
//! Maintains a single-layer proximity graph that supports post-build
//! inserts and logical deletes without full rebuilds. Deleted nodes are marked
//! with tombstones: search traverses through them (to preserve graph
//! connectivity) but excludes them from result sets. Compaction removes
//! tombstones and remaps neighbor IDs.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.8", features = ["fresh_graph"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::fresh_graph::{FreshGraphIndex, FreshGraphParams};
//!
//! let params = FreshGraphParams::default();
//! let mut index = FreshGraphIndex::new(128, params)?;
//!
//! for (id, vec) in data {
//!     index.add(id, vec)?;
//! }
//! index.build()?;
//!
//! // Insert new vectors without rebuild
//! index.insert(new_id, &new_vec)?;
//!
//! // Delete by doc_id (logical tombstone)
//! index.delete(old_id)?;
//!
//! let results = index.search(&query, 10)?;
//!
//! // Compact when tombstone ratio is high
//! index.compact()?;
//! ```
//!
//! # Design
//!
//! Construction follows NSG/Vamana style: compute medoid as entry point,
//! build a brute-force kNN graph for small n or random-init for large n,
//! then apply RNG (Relative Neighborhood Graph) pruning with a relaxation
//! factor alpha. Post-build inserts use beam search to find the best
//! neighborhood, prune with the same RNG criterion, and add reverse edges
//! with degree cap. Deleted nodes stay in the graph structure so search
//! can traverse through them; they are simply excluded from returned results.
//!
//! # References
//!
//! - Singh et al. (2021). "FreshDiskANN: A Fast and Accurate Graph-Based ANN
//!   Index for Streaming Similarity Search."
//! - Xu, Bernstein, Guestrin (2023). "CleANN: Efficient Concurrent Insertions
//!   and Deletions for Graph-based ANN indexes." arXiv:2310.03264.

use crate::distance::cosine_distance_normalized;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::collections::BinaryHeap;

/// FreshGraph construction and search parameters.
#[derive(Clone, Debug)]
pub struct FreshGraphParams {
    /// Maximum out-degree per node. Default: 32.
    pub max_degree: usize,
    /// Candidate pool size during construction. Default: 200.
    pub ef_construction: usize,
    /// Candidate pool size during search. Default: 100.
    pub ef_search: usize,
    /// RNG relaxation factor (alpha >= 1.0). Higher = denser graph. Default: 1.2.
    pub alpha: f32,
}

impl Default for FreshGraphParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            ef_construction: 200,
            ef_search: 100,
            alpha: 1.2,
        }
    }
}

/// FreshGraph index.
///
/// Uses **cosine distance only**. Vectors are L2-normalized in
/// [`Self::add_slice`] / [`Self::insert`]; callers do not need to
/// pre-normalize. Pre-normalized input is also fine (the second
/// normalization is a no-op up to f32 rounding). This module does not
/// honor the [`crate::distance::DistanceMetric`] enum; for L2 or inner
/// product on a streaming graph, use a different index.
pub struct FreshGraphIndex {
    dimension: usize,
    params: FreshGraphParams,

    /// Flat normalized vector storage (row-major, `dimension` floats per vector).
    vectors: Vec<f32>,
    num_vectors: usize,
    doc_ids: Vec<u32>,

    /// Per-node adjacency lists (internal IDs).
    neighbors: Vec<SmallVec<[u32; 16]>>,

    /// In-edge count per node, maintained incrementally across `add_slice`,
    /// `build`, `insert`, and `compact`. Drives the orphan-protection check
    /// in `add_reverse_edge_protected` (invariant I2). Edges from tombstoned
    /// nodes still count: search traverses through tombstones, so those
    /// in-edges are real for reachability purposes until `compact` removes
    /// the source nodes and rebuilds the count.
    inbound_count: Vec<u32>,

    /// Tombstone flags. True = deleted (traversed but excluded from results).
    deleted: Vec<bool>,
    num_deleted: usize,

    /// Entry point for beam search (medoid of the initial batch).
    entry_point: u32,
    built: bool,
}

impl FreshGraphIndex {
    /// Create a new FreshGraph index with the given dimension and parameters.
    pub fn new(dimension: usize, params: FreshGraphParams) -> Result<Self, RetrieveError> {
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
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            neighbors: Vec::new(),
            inbound_count: Vec::new(),
            deleted: Vec::new(),
            num_deleted: 0,
            entry_point: 0,
            built: false,
        })
    }

    /// Add a vector to the pre-build staging area (before `build`).
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector from a slice to the pre-build staging area.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "use insert() to add vectors after build".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }
        self.store_normalized(vector);
        self.doc_ids.push(doc_id);
        self.deleted.push(false);
        self.num_vectors += 1;
        Ok(())
    }

    /// Recompute `inbound_count` from scratch over the current neighbor lists.
    /// O(n × avg_degree). Used after wholesale graph mutations
    /// (`build_knn_graph_*`, `compact`, `ensure_connectivity`) where
    /// incremental tracking would be more invasive than the one-shot cost.
    fn recompute_inbound_count(&mut self) {
        self.inbound_count = vec![0; self.num_vectors];
        for nbrs in &self.neighbors {
            for &id in nbrs {
                if (id as usize) < self.inbound_count.len() {
                    self.inbound_count[id as usize] += 1;
                }
            }
        }
    }

    /// Build the initial graph from staged vectors.
    ///
    /// After `build`, use `insert` for new vectors and `delete` for removals.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let n = self.num_vectors;

        // Compute medoid and use it as the entry point
        self.entry_point = self.compute_medoid();

        // Build initial kNN graph: brute-force for small n, random-init for large
        const BRUTE_FORCE_THRESHOLD: usize = 2000;
        if n <= BRUTE_FORCE_THRESHOLD {
            self.build_knn_graph_brute_force();
        } else {
            self.build_knn_graph_random();
        }

        // Initialize inbound_count from the kNN graph laid down by
        // build_knn_graph_*. After this point the helper maintains it
        // incrementally as edges are added or evicted, and the per-row
        // forward-edge swap below adjusts it explicitly.
        self.recompute_inbound_count();

        // RNG pruning pass: for each node, beam search for candidates then prune.
        // Reverse edges go through `add_reverse_edge_protected` so that nodes
        // whose only remaining inbound edge would be the one through this
        // forward neighbor are kept (orphan protection -- invariant I2 of the
        // delete-reinsert reachability test).
        let mut inbound_count = std::mem::take(&mut self.inbound_count);
        for i in 0..n {
            let vi = self.get_vector(i).to_vec();
            let candidates = self.beam_search_internal(&vi, self.params.ef_construction, None);
            let selected = self.rng_prune(&vi, &candidates);

            // Replacing self.neighbors[i] mutates the inbound counts of the
            // old and new forward neighbors. Apply those deltas before the
            // reverse-edge loop so the protection check sees a consistent
            // count when it inspects neighbors[nid] further down.
            let old_neighbors = std::mem::replace(
                &mut self.neighbors[i],
                selected.iter().map(|&(id, _)| id).collect(),
            );
            for &old in &old_neighbors {
                if (old as usize) < inbound_count.len() && inbound_count[old as usize] > 0 {
                    inbound_count[old as usize] -= 1;
                }
            }
            for (id, _) in &selected {
                if (*id as usize) < inbound_count.len() {
                    inbound_count[*id as usize] += 1;
                }
            }

            let max_deg = self.params.max_degree;
            for &(neighbor_id, _) in &selected {
                self.add_reverse_edge_protected(neighbor_id, i as u32, max_deg, &mut inbound_count);
            }

            drop(old_neighbors);
        }
        self.inbound_count = inbound_count;

        // Ensure all nodes are reachable from the entry point. The bridge
        // edges this adds are not tracked through the helper, so the
        // count is recomputed once after.
        self.ensure_connectivity();
        self.recompute_inbound_count();

        self.built = true;
        Ok(())
    }

    /// Insert a new vector into the built index without a full rebuild.
    ///
    /// Uses beam search from the entry point to find the best neighborhood,
    /// applies RNG pruning, inserts bidirectional edges, and updates the entry
    /// point if the new vector is a better medoid.
    pub fn insert(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "call build() before insert()".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        self.store_normalized(vector);
        let new_id = self.num_vectors as u32;
        self.doc_ids.push(doc_id);
        self.deleted.push(false);
        self.num_vectors += 1;
        self.neighbors.push(SmallVec::new());
        // Track the new node in inbound_count starting at 0; the forward-
        // and reverse-edge phases below update it as edges are written.
        self.inbound_count.push(0);

        let new_vec = self.get_vector(new_id as usize).to_vec();

        // Beam search to find candidate neighbors (skip deleted in results but
        // traverse through them -- beam_search_internal handles this via the
        // skip_in_results flag)
        let candidates =
            self.beam_search_internal(&new_vec, self.params.ef_construction, Some(new_id));

        // RNG prune the candidate list
        let selected = self.rng_prune(&new_vec, &candidates);

        // Set new node's neighbor list (forward edges) and increment the
        // inbound_count of every forward neighbor that just gained an
        // in-edge from new_id.
        self.neighbors[new_id as usize] = selected.iter().map(|&(id, _)| id).collect();
        for &(f, _) in &selected {
            if (f as usize) < self.inbound_count.len() {
                self.inbound_count[f as usize] += 1;
            }
        }

        // Reverse-edge phase. The helper takes `inbound_count` by `&mut [u32]`
        // so it can update counts as it appends or evicts; mem::take satisfies
        // the borrow checker without copying.
        let max_deg = self.params.max_degree;
        let mut inbound_count = std::mem::take(&mut self.inbound_count);
        for &(neighbor_id, _) in &selected {
            self.add_reverse_edge_protected(neighbor_id, new_id, max_deg, &mut inbound_count);
        }
        self.inbound_count = inbound_count;

        Ok(())
    }

    /// Add a reverse edge `nid <- new_id` with degree-cap eviction that
    /// preserves graph reachability.
    ///
    /// Three branches:
    /// 1. `nid` already lists `new_id`: nothing to do.
    /// 2. `nid` is below `max_deg`: append `new_id`; increment its
    ///    inbound count.
    /// 3. `nid` is at `max_deg`: orphan-protected re-prune. Candidates
    ///    are `neighbors[nid] ∪ {new_id}`. A candidate whose
    ///    `inbound_count <= 1` would become unreachable if evicted
    ///    (its only in-edge is the one through `nid`); those are kept
    ///    unconditionally. RNG-prune chooses among the unprotected
    ///    remainder for the remaining slots. If protected candidates
    ///    overflow `max_deg`, fall back to plain RNG-prune over all
    ///    (the graph is already pathologically sparse and can't honor
    ///    every protection).
    ///
    /// `inbound_count` is updated in place.
    fn add_reverse_edge_protected(
        &mut self,
        neighbor_id: u32,
        new_id: u32,
        max_deg: usize,
        inbound_count: &mut [u32],
    ) {
        let nid = neighbor_id as usize;
        if self.neighbors[nid].contains(&new_id) {
            return;
        }
        if self.neighbors[nid].len() < max_deg {
            self.neighbors[nid].push(new_id);
            if (new_id as usize) < inbound_count.len() {
                inbound_count[new_id as usize] += 1;
            }
            return;
        }

        let nv = self.get_vector(nid).to_vec();
        let mut rev_cands: Vec<(u32, f32, bool)> = self.neighbors[nid]
            .iter()
            .copied()
            .chain(std::iter::once(new_id))
            .map(|id| {
                let d = cosine_distance_normalized(&nv, self.get_vector(id as usize));
                let is_orphan_if_evicted =
                    (id as usize) < inbound_count.len() && inbound_count[id as usize] <= 1;
                (id, d, is_orphan_if_evicted)
            })
            .collect();

        let mut keepers: Vec<(u32, f32)> = rev_cands
            .iter()
            .filter(|(_, _, orphan)| *orphan)
            .map(|&(id, d, _)| (id, d))
            .collect();

        if keepers.len() > max_deg {
            rev_cands.sort_by(|a, b| a.1.total_cmp(&b.1));
            let flat: Vec<(u32, f32)> = rev_cands.iter().map(|&(id, d, _)| (id, d)).collect();
            let pruned = self.rng_prune(&nv, &flat);
            keepers = pruned.iter().map(|&(id, d)| (id, d)).collect();
        } else {
            let remaining_slots = max_deg - keepers.len();
            if remaining_slots > 0 {
                let unprotected: Vec<(u32, f32)> = rev_cands
                    .iter()
                    .filter(|(_, _, orphan)| !*orphan)
                    .map(|&(id, d, _)| (id, d))
                    .collect();
                let pruned = self.rng_prune(&nv, &unprotected);
                for (id, d) in pruned.into_iter().take(remaining_slots) {
                    keepers.push((id, d));
                }
            }
        }

        let new_set: std::collections::HashSet<u32> = keepers.iter().map(|(id, _)| *id).collect();
        let old_set: std::collections::HashSet<u32> = self.neighbors[nid].iter().copied().collect();
        for &evicted in old_set.difference(&new_set) {
            if (evicted as usize) < inbound_count.len() && inbound_count[evicted as usize] > 0 {
                inbound_count[evicted as usize] -= 1;
            }
        }
        for &added in new_set.difference(&old_set) {
            if (added as usize) < inbound_count.len() {
                inbound_count[added as usize] += 1;
            }
        }
        self.neighbors[nid] = keepers.iter().map(|(id, _)| *id).collect();
    }

    /// Mark a vector as deleted by its doc_id (logical tombstone).
    ///
    /// Returns `Ok(true)` if the doc_id was found and deleted, `Ok(false)` if
    /// not found. Deleted nodes remain in the graph for traversal but are
    /// excluded from search results.
    pub fn delete(&mut self, doc_id: u32) -> Result<bool, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "call build() before delete()".into(),
            ));
        }
        for i in 0..self.num_vectors {
            if self.doc_ids[i] == doc_id && !self.deleted[i] {
                self.deleted[i] = true;
                self.num_deleted += 1;

                // If the deleted node was the entry point, promote a live
                // neighbor (preferred) or any live node (fallback) so future
                // searches and inserts start from a non-tombstoned anchor.
                // Without this, repeated delete-reinsert cycles that happen
                // to victimize the medoid leave the search beam rooted in a
                // tombstone whose neighbor list points at a subgraph the
                // post-cycle live set may have drifted away from.
                if self.entry_point == i as u32 {
                    self.repromote_entry_point();
                }

                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Pick a new entry point after the previous one was deleted. Prefer a
    /// live neighbor of the now-tombstoned entry (preserves graph locality);
    /// otherwise scan for any live node.
    fn repromote_entry_point(&mut self) {
        let stale = self.entry_point as usize;

        // Try the stale entry's own neighbors first; one of them is almost
        // certainly still live and well-connected.
        for &candidate in &self.neighbors[stale] {
            if !self.deleted[candidate as usize] {
                self.entry_point = candidate;
                return;
            }
        }

        // Fall back to a linear scan. This costs O(n) but only fires when
        // the deleted entry's whole neighborhood is also tombstoned, which
        // means the graph is already badly degraded.
        if let Some(alt) = (0..self.num_vectors as u32).find(|&j| !self.deleted[j as usize]) {
            self.entry_point = alt;
        }
        // If even the linear scan fails, every node is deleted; leave
        // entry_point alone. search() / insert() will handle an empty live
        // set higher up.
    }

    /// Search for the k nearest neighbors of a query vector.
    ///
    /// Deleted nodes are traversed for graph connectivity but excluded from
    /// the returned result set.
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

        let query_normalized = self.normalize(query);
        let ef = self.params.ef_search.max(k);

        // Beam search returns internal IDs sorted by distance
        let candidates = self.beam_search_internal(&query_normalized, ef, None);

        // Filter deleted, map to doc_ids, take k
        let results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|&(id, _)| !self.deleted[id as usize])
            .take(k)
            .map(|(id, dist)| (self.doc_ids[id as usize], dist))
            .collect();

        Ok(results)
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

        let query_normalized = self.normalize(query);
        let ef = ef_search.max(k);

        let candidates = self.beam_search_internal(&query_normalized, ef, None);

        let results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|&(id, _)| !self.deleted[id as usize])
            .take(k)
            .map(|(id, dist)| (self.doc_ids[id as usize], dist))
            .collect();

        Ok(results)
    }

    /// Remove all tombstoned vectors, remap neighbor IDs, and rebuild
    /// connectivity. Does nothing if there are no deleted nodes.
    #[allow(clippy::needless_range_loop)]
    pub fn compact(&mut self) -> Result<(), RetrieveError> {
        if self.num_deleted == 0 {
            return Ok(());
        }

        let old_n = self.num_vectors;

        // Build remap: old internal ID -> new internal ID (None = deleted)
        let mut remap: Vec<Option<u32>> = vec![None; old_n];
        let mut new_id = 0u32;
        for i in 0..old_n {
            if !self.deleted[i] {
                remap[i] = Some(new_id);
                new_id += 1;
            }
        }
        let new_n = new_id as usize;

        if new_n == 0 {
            // All vectors were deleted -- reset to empty pre-build state
            self.vectors.clear();
            self.num_vectors = 0;
            self.doc_ids.clear();
            self.neighbors.clear();
            self.inbound_count.clear();
            self.deleted.clear();
            self.num_deleted = 0;
            self.entry_point = 0;
            self.built = false;
            return Ok(());
        }

        // Compact vector storage
        let dim = self.dimension;
        let mut new_vectors = Vec::with_capacity(new_n * dim);
        let mut new_doc_ids = Vec::with_capacity(new_n);
        let mut new_neighbors: Vec<SmallVec<[u32; 16]>> = Vec::with_capacity(new_n);

        for i in 0..old_n {
            if !self.deleted[i] {
                let start = i * dim;
                new_vectors.extend_from_slice(&self.vectors[start..start + dim]);
                new_doc_ids.push(self.doc_ids[i]);

                // Remap neighbor IDs, dropping references to deleted nodes
                let remapped: SmallVec<[u32; 16]> = self.neighbors[i]
                    .iter()
                    .filter_map(|&nb| remap[nb as usize])
                    .collect();
                new_neighbors.push(remapped);
            }
        }

        // Update entry point
        let new_entry = remap[self.entry_point as usize].unwrap_or(0);

        self.vectors = new_vectors;
        self.num_vectors = new_n;
        self.doc_ids = new_doc_ids;
        self.neighbors = new_neighbors;
        self.deleted = vec![false; new_n];
        self.num_deleted = 0;
        self.entry_point = new_entry;

        // Re-establish connectivity lost when deleted nodes were removed.
        // ensure_connectivity may add bridge edges that bypass the helper,
        // so the count is recomputed once after both steps complete.
        self.ensure_connectivity();
        self.recompute_inbound_count();

        Ok(())
    }

    /// Number of vectors in the index (including tombstoned ones).
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index has no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Number of logically deleted (tombstoned) vectors.
    pub fn num_deleted(&self) -> usize {
        self.num_deleted
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /// Normalize and store a vector.
    #[inline]
    fn store_normalized(&mut self, vector: &[f32]) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            self.vectors.extend(vector.iter().map(|x| x / norm));
        } else {
            self.vectors.extend_from_slice(vector);
        }
    }

    /// Return a normalized copy of `v`.
    #[inline]
    fn normalize(&self, v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    /// Compute the medoid (node closest to the centroid of all live vectors).
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
            if self.deleted[i] {
                continue;
            }
            let d = cosine_distance_normalized(&centroid, self.get_vector(i));
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
    }

    /// Build a brute-force kNN graph (O(n^2), suitable for n <= ~2000).
    fn build_knn_graph_brute_force(&mut self) {
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

    /// Build an approximate kNN graph by scanning a random window of neighbors.
    /// Used for large n where brute-force is impractical.
    fn build_knn_graph_random(&mut self) {
        let n = self.num_vectors;
        let k = self.params.max_degree.min(n.saturating_sub(1));
        let scan = (k * 4).min(n);
        self.neighbors = vec![SmallVec::new(); n];

        for i in 0..n {
            let vi = self.get_vector(i);
            let mut dists: Vec<(u32, f32)> = (0..scan)
                .filter(|&j| j != i)
                .map(|j| (j as u32, cosine_distance_normalized(vi, self.get_vector(j))))
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(k);
            self.neighbors[i] = dists.iter().map(|(id, _)| *id).collect();
        }
    }

    /// RNG pruning with alpha relaxation.
    ///
    /// For a candidate c, keep it if no already-selected neighbor s satisfies:
    ///   alpha * d(query, s) >= d(s, c)
    ///
    /// With alpha=1.0 this is strict MRNG. alpha>1.0 relaxes the condition,
    /// allowing more edges and denser graphs.
    fn rng_prune(&self, _query_vec: &[f32], candidates: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
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
                // RNG alpha condition: prune c if alpha * d(q,s) >= d(s,c)
                // Equivalently: keep c if d(s,c) > alpha * d(q,s) for all s
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

    /// Beam search from the entry point.
    ///
    /// `exclude`: an internal ID to exclude from the candidate set (used
    /// during insert to avoid a node listing itself as its own neighbor).
    ///
    /// Deleted nodes are traversed (for connectivity) but can be filtered
    /// from results by callers.
    fn beam_search_internal(
        &self,
        query: &[f32],
        ef: usize,
        exclude: Option<u32>,
    ) -> Vec<(u32, f32)> {
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

            // Min-heap on distance
            let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
            let mut candidates: Vec<(u32, f32)> = Vec::new();

            let entry = self.entry_point;
            if exclude == Some(entry) {
                // Entry point is the new node itself; pick its first neighbor or node 0
                // Fallback: find a non-excluded, non-deleted entry among all nodes
                if let Some(alt) =
                    (0..n as u32).find(|&i| Some(i) != exclude && !self.deleted[i as usize])
                {
                    let d = cosine_distance_normalized(query, self.get_vector(alt as usize));
                    visited_insert(alt);
                    frontier.push(std::cmp::Reverse((FloatOrd(d), alt)));
                    candidates.push((alt, d));
                } else {
                    return Vec::new();
                }
            } else {
                let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
                visited_insert(entry);
                frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
                candidates.push((entry, entry_dist));
            }

            let mut visited_count = candidates.len();

            while let Some(std::cmp::Reverse((FloatOrd(current_dist), current_id))) = frontier.pop() {
                // Early termination: current node is worse than the ef-th best candidate
                if candidates.len() >= ef {
                    candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    if current_dist > candidates[ef - 1].1 * 1.5 {
                        break;
                    }
                }

                let neighbors = &self.neighbors[current_id as usize];
                for (i, &neighbor) in neighbors.iter().enumerate() {
                    if Some(neighbor) == exclude {
                        continue;
                    }

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
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.entry_point, |i, j| {
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

    fn default_params() -> FreshGraphParams {
        FreshGraphParams {
            max_degree: 16,
            ef_construction: 50,
            ef_search: 50,
            alpha: 1.2,
        }
    }

    #[test]
    fn build_and_search() {
        let dim = 16;
        let n = 40;
        let data = make_vectors(n, dim, 42);

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        let results = index.search(&data[0..dim], 5).unwrap();
        assert!(!results.is_empty());
        assert!(
            results.iter().any(|(id, _)| *id == 0),
            "expected doc 0 in top-5: {:?}",
            results
        );
    }

    #[test]
    fn insert_after_build() {
        let dim = 16;
        let n = 30;
        let data = make_vectors(n + 1, dim, 99);

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        // Insert one more vector
        let new_vec = &data[n * dim..(n + 1) * dim];
        index.insert(n as u32, new_vec).unwrap();

        assert_eq!(index.len(), n + 1);

        // The inserted vector should appear in a self-search
        let results = index.search(new_vec, 3).unwrap();
        assert!(
            results.iter().any(|(id, _)| *id == n as u32),
            "inserted doc not found: {:?}",
            results
        );
    }

    #[test]
    fn delete_removes_from_results() {
        let dim = 16;
        let n = 40;
        let data = make_vectors(n, dim, 7);

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        // Confirm doc 0 is found before deletion
        let before = index.search(&data[0..dim], 5).unwrap();
        assert!(before.iter().any(|(id, _)| *id == 0));

        // Delete it
        let found = index.delete(0).unwrap();
        assert!(found);

        assert_eq!(index.num_deleted(), 1);

        // Doc 0 must no longer appear in results
        let after = index.search(&data[0..dim], 5).unwrap();
        assert!(
            !after.iter().any(|(id, _)| *id == 0),
            "deleted doc 0 still in results: {:?}",
            after
        );
    }

    #[test]
    fn delete_nonexistent_returns_false() {
        let dim = 8;
        let n = 10;
        let data = make_vectors(n, dim, 55);

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        let found = index.delete(9999).unwrap();
        assert!(!found);
    }

    #[test]
    fn compact_recovers_space() {
        let dim = 16;
        let n = 40;
        let data = make_vectors(n, dim, 13);

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..n {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        // Delete half the vectors
        for i in 0..(n / 2) {
            index.delete(i as u32).unwrap();
        }
        assert_eq!(index.num_deleted(), n / 2);

        index.compact().unwrap();

        // After compact, num_deleted is 0 and len drops
        assert_eq!(index.num_deleted(), 0);
        assert_eq!(index.len(), n / 2);

        // inbound_count must be consistent with the rebuilt neighbor lists.
        // compact remaps neighbor IDs and calls recompute_inbound_count;
        // a regression in the remap loop (e.g. dropping a neighbor reference)
        // would leave the count under-reporting in-edges and re-fire the
        // orphan-protection branch incorrectly on subsequent inserts.
        {
            let n_post = index.num_vectors;
            let mut actual = vec![0u32; n_post];
            for adj in &index.neighbors {
                for &id in adj {
                    if (id as usize) < n_post {
                        actual[id as usize] += 1;
                    }
                }
            }
            for v in 0..n_post {
                assert_eq!(
                    index.inbound_count[v], actual[v],
                    "post-compact inbound_count[{v}]={} but actual in-edges={}",
                    index.inbound_count[v], actual[v],
                );
            }
        }

        // Search still works for surviving docs
        let query = &data[(n / 2) * dim..(n / 2 + 1) * dim];
        let results = index.search(query, 3).unwrap();
        assert!(!results.is_empty());

        // Deleted docs must not appear
        for (id, _) in &results {
            assert!(
                *id >= (n / 2) as u32,
                "deleted doc {id} appeared after compact"
            );
        }
    }

    #[test]
    fn self_search_recall() {
        let dim = 16;
        let n = 50;
        let data = make_vectors(n, dim, 77);

        let mut index = FreshGraphIndex::new(
            dim,
            FreshGraphParams {
                max_degree: 16,
                ef_construction: 80,
                ef_search: 80,
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

        let mut hits = 0;
        for i in 0..n {
            let results = index.search(&data[i * dim..(i + 1) * dim], 1).unwrap();
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
    fn empty_index_errors() {
        let mut index = FreshGraphIndex::new(8, FreshGraphParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn add_after_build_errors() {
        let dim = 8;
        let data = make_vectors(5, dim, 1);

        let mut index = FreshGraphIndex::new(dim, FreshGraphParams::default()).unwrap();
        for i in 0..5usize {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();

        // add_slice should fail after build
        assert!(index.add_slice(99, &data[0..dim]).is_err());
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut index = FreshGraphIndex::new(8, FreshGraphParams::default()).unwrap();
        assert!(index.add(0, vec![1.0; 16]).is_err());
    }

    /// Static post-build connectivity check: every non-entry node has at
    /// least one inbound edge in the layer-0 graph. This is the static
    /// analog of I1 -- it doesn't catch the dynamic delete-reinsert bug
    /// the 200-cycle test guards against, but it does catch build-time
    /// orphans that would silently degrade recall without manifesting in
    /// self-search of the queried node.
    ///
    /// The qdrant project ships an analogous `test_graph_connectivity` for
    /// HNSW; this test ports the pattern to FreshGraph using the now-
    /// maintained `inbound_count` field directly. The entry point itself
    /// may legitimately have zero inbound edges (search is rooted there;
    /// no other node needs to point at it for reachability).
    #[test]
    fn build_post_state_every_non_entry_node_has_inbound_edge() {
        // Walk multiple seeds: a single-seed connectivity test can pass on
        // a layout that happens to leave no orphans even when the underlying
        // invariants are weakened. Five seeds give a wider sample without
        // appreciable runtime cost (n=100, build is cheap).
        let dim = 16;
        let n = 100;
        for &seed in &[0xABCDu64, 0x1234, 0xDEADBEEF, 0x42, 0xC0FFEE] {
            let data = make_vectors(n, dim, seed);

            let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
            for i in 0..n {
                index
                    .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                    .unwrap();
            }
            index.build().unwrap();

            let entry = index.entry_point;
            let mut orphans: Vec<u32> = Vec::new();
            for i in 0..n as u32 {
                if i == entry {
                    continue;
                }
                if index.inbound_count[i as usize] == 0 {
                    orphans.push(i);
                }
            }

            assert!(
                orphans.is_empty(),
                "seed {seed:#x}: {} of {} non-entry nodes have zero inbound edges \
                 after build: {:?}. likely cause: ensure_connectivity failed to \
                 bridge an isolated component, or the RNG-prune evicted the only \
                 remaining in-edge to these nodes (invariant I2 broken at build \
                 time, not just after delete-reinsert cycles).",
                orphans.len(),
                n - 1,
                &orphans[..orphans.len().min(10)],
            );
        }
    }

    /// Direct invariant guard for `inbound_count`: for every node `v`, the
    /// stored count must equal the number of forward edges in the graph that
    /// point at `v`. The connectivity test above only catches drift once it
    /// pushes a count to zero (orphans); this test catches drift in either
    /// direction (over- or under-count) at every node, which would silently
    /// corrupt the orphan-protection branch in `add_reverse_edge_protected`.
    ///
    /// Edges from tombstoned source nodes count as real in-edges -- the doc
    /// comment on the field documents this, and beam search traverses through
    /// tombstones. We exercise build, then a sequence of `insert` and
    /// `delete`, asserting consistency at each step so a regression points
    /// at the exact operation that broke it.
    #[test]
    fn inbound_count_matches_forward_edges_through_inserts_and_deletes() {
        fn assert_consistent(index: &FreshGraphIndex, ctx: &str) {
            let n = index.num_vectors;
            let mut actual = vec![0u32; n];
            for adj in &index.neighbors {
                for &id in adj {
                    if (id as usize) < n {
                        actual[id as usize] += 1;
                    }
                }
            }
            for v in 0..n {
                assert_eq!(
                    index.inbound_count[v], actual[v],
                    "{ctx}: inbound_count[{v}]={} but actual in-edge count={} \
                     (drift in incremental tracking; fresh recompute would have \
                     produced {})",
                    index.inbound_count[v], actual[v], actual[v],
                );
            }
        }

        let dim = 8;
        let total = 50;
        let data = make_vectors(total, dim, 0xC0DE);

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..30 {
            index
                .add_slice(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
        }
        index.build().unwrap();
        assert_consistent(&index, "after build");

        for i in 30..total {
            index
                .insert(i as u32, &data[i * dim..(i + 1) * dim])
                .unwrap();
            assert_consistent(&index, &format!("after insert doc_id={i}"));
        }

        // Delete a spread of doc_ids covering the build-time and post-build
        // ranges, the entry point neighborhood, and a few late inserts.
        for &doc_id in &[1u32, 5, 12, 18, 22, 28, 33, 40, 44, 49] {
            index.delete(doc_id).unwrap();
            assert_consistent(&index, &format!("after delete doc_id={doc_id}"));
        }
    }

    /// Manual scaling probe for orphan-protection cost in `insert()`.
    ///
    /// Two costs interact in `insert()`'s reverse-edge phase:
    ///
    /// 1. The orphan-protected RNG-prune in `add_reverse_edge_protected`
    ///    fires once per forward neighbor (up to `max_degree` times),
    ///    each call doing O(max_degree²) distance work. This dominates
    ///    at small n once every node is at cap.
    ///
    /// 2. `inbound_count` was previously rebuilt from scratch per call at
    ///    `O(n × max_degree)`. As of v0.7.2 it is maintained as a struct
    ///    field across calls, so this term drops out.
    ///
    /// Together: per-insert cost was `O(max_degree²) + O(n × max_degree)`
    /// and is now `O(max_degree²)` — flat in n once the at-cap regime is
    /// reached. The probe walks `n` past the prior 2000-cap so the
    /// flatness is visible (without incremental tracking the per-op cost
    /// would grow linearly with n past the cap-saturation point).
    ///
    /// Marked `#[ignore]` because it's a measurement, not a regression
    /// guard. Run with:
    ///   cargo test --release --features fresh_graph
    ///       fresh_graph::tests::orphan_protection_scaling_probe
    ///       -- --ignored --nocapture
    #[test]
    #[ignore = "measurement only; run with --release --ignored --nocapture"]
    fn orphan_protection_scaling_probe() {
        use std::time::Instant;

        println!("# orphan-protection scaling probe");
        println!("# n, cycles, dim, build_ms, cycle_total_ms, cycle_us_per_op");

        for &n in &[100usize, 500, 1000, 2000, 5000, 10000, 25000, 50000] {
            let dim = 32;
            let cycles = (n / 4).clamp(50, 500);
            let extra_pool = cycles + n;
            let data = make_vectors(extra_pool, dim, 0xC0FFEE);
            let vec_at = |id: u32| -> &[f32] { &data[id as usize * dim..(id as usize + 1) * dim] };

            let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
            for i in 0..n as u32 {
                index.add_slice(i, vec_at(i)).unwrap();
            }
            let t_build = Instant::now();
            index.build().unwrap();
            let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

            let mut active: Vec<u32> = (0..n as u32).collect();
            let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_F00D;
            let t_cycle = Instant::now();
            for new_id in (n as u32..).take(cycles) {
                rng_state = rng_state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let victim_pos = ((rng_state >> 33) as usize) % active.len().max(1);
                let victim = active.swap_remove(victim_pos);
                index.delete(victim).unwrap();
                index.insert(new_id, vec_at(new_id)).unwrap();
                active.push(new_id);
            }
            let cycle_total_ms = t_cycle.elapsed().as_secs_f64() * 1000.0;
            let cycle_us_per_op = (cycle_total_ms * 1000.0) / cycles as f64;
            println!("{n},{cycles},{dim},{build_ms:.1},{cycle_total_ms:.1},{cycle_us_per_op:.1}");
        }
    }

    /// Regression guard for delete-then-reinsert cycles in FreshGraph.
    ///
    /// arxiv:2407.07871 describes naive HNSW variants accumulating
    /// unreachable nodes after delete-reinsert sequences (3-4% after
    /// ~3000 cycles in their setup). FreshGraph holds two invariants
    /// that together close that failure mode:
    ///
    /// **I1. Entry-point liveness.** `entry_point` always references a
    /// non-tombstoned internal id (or the index has no live nodes at
    /// all). `delete()` repromotes when it tombstones the current
    /// entry; without this, search and insert root their beam at a
    /// dead anchor whose neighbor list points at a subgraph the live
    /// set may have drifted away from.
    ///
    /// **I2. In-edge survival under reverse-edge prune.** When
    /// `insert()` adds a reverse edge to a forward neighbor `F` that
    /// is already at `max_degree`, the prune step never evicts a node
    /// whose only remaining inbound edge would be the one through `F`.
    /// RNG-prune runs only over candidates with redundancy elsewhere
    /// (inbound count > 1). Without this, the reverse-edge handler
    /// can evict any long-existing edge with no preference for
    /// connectivity, so an original id whose other in-edges have been
    /// pruned away in earlier cycles becomes unreachable in this one.
    ///
    /// Concretely: this test runs 200 delete-reinsert cycles on a
    /// 60-vector FreshGraph and asserts every live doc_id is reachable
    /// by self-search. If either invariant breaks (e.g., the entry-
    /// point repromotion is removed, or the orphan-protection branch
    /// in `insert`'s reverse-edge prune is bypassed), the test reports
    /// the unreachable ids and the assertion message names the
    /// invariant for the reader to look up.
    #[test]
    fn delete_reinsert_cycles_preserve_reachability() {
        let dim = 16;
        let n = 60;
        let cycles = 200;
        let extra_pool = cycles + n;
        let data = make_vectors(extra_pool, dim, 0xC0FFEE);
        let vec_at = |id: u32| &data[id as usize * dim..(id as usize + 1) * dim];

        let mut index = FreshGraphIndex::new(dim, default_params()).unwrap();
        for i in 0..n as u32 {
            index.add_slice(i, vec_at(i)).unwrap();
        }
        index.build().unwrap();

        let mut active: Vec<u32> = (0..n as u32).collect();
        // new doc_ids start at n and count up monotonically as we cycle.

        // Deterministic LCG so failures bisect.
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        let rand_index = |bound: usize, state: &mut u64| -> usize {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((*state >> 33) as usize) % bound.max(1)
        };

        for new_id in (n as u32..).zip(0..cycles).map(|(id, _)| id) {
            let victim_pos = rand_index(active.len(), &mut rng_state);
            let victim = active.swap_remove(victim_pos);
            assert!(index.delete(victim).unwrap());

            // Re-insert with a fresh doc_id and a fresh vector slot.
            assert!(new_id < extra_pool as u32);
            index.insert(new_id, vec_at(new_id)).unwrap();
            active.push(new_id);
        }

        // After all cycles, every live doc_id should still be reachable
        // by searching for its own vector.
        let mut missing: Vec<u32> = Vec::new();
        for &id in &active {
            let v = vec_at(id);
            // ef_search high enough that "missing" really means unreachable,
            // not just "didn't fit in the small candidate pool".
            let results = index
                .search_with_ef(v, 5, 200)
                .expect("search after cycles");
            if !results.iter().any(|(rid, _)| *rid == id) {
                missing.push(id);
            }
        }

        assert!(
            missing.is_empty(),
            "{} of {} live ids became unreachable after {} delete-reinsert cycles: {:?}. \
             likely cause: invariant I1 (entry-point liveness in delete) or I2 \
             (in-edge survival in insert reverse-edge prune) was broken. \
             see this test's docstring for what each invariant guarantees.",
            missing.len(),
            active.len(),
            cycles,
            &missing[..missing.len().min(10)]
        );
    }
}
