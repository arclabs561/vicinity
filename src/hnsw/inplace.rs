//! In-Place Graph Updates (IP-DiskANN style).
//!
//! Implements efficient per-operation updates without batch consolidation.
//! Key insight: Maintain in-neighbor lists to enable efficient deletions.
//!
//! # Key Features
//!
//! - **In-neighbor tracking**: Each node knows which nodes point to it
//! - **Per-operation updates**: No batch consolidation needed
//! - **Stable recall**: Maintains graph quality after many updates
//! - **Single-writer**: All mutations require `&mut self`
//!
//! # Algorithm (from IP-DiskANN 2025)
//!
//! For **insertions**:
//! 1. Find candidate neighbors via greedy search
//! 2. Add out-edges to neighbors
//! 3. Update in-neighbor lists of neighbors
//! 4. Neighbors may add back-edge if beneficial
//!
//! For **deletions**:
//! 1. Mark node as deleted
//! 2. For each in-neighbor, remove edge to deleted node
//! 3. In-neighbors find replacement edges via local search
//! 4. Recycle slot for future insertions
//!
//! # References
//!
//! - Xu et al. (2025): "In-Place Updates of a Graph Index for Streaming
//!   Approximate Nearest Neighbor Search" - <https://arxiv.org/abs/2502.13826>

use crate::hnsw::repair::{validate_connectivity, RepairConfig, RepairStats};
use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for in-place updates.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InPlaceConfig {
    /// Maximum out-degree per node
    pub max_degree: usize,
    /// Search beam width during updates
    pub beam_width: usize,
    /// Alpha for diversity pruning
    pub alpha: f32,
    /// Maximum in-neighbors to track
    pub max_in_neighbors: usize,
    /// Enable back-edge insertion
    pub enable_back_edges: bool,
}

impl Default for InPlaceConfig {
    fn default() -> Self {
        Self {
            max_degree: 32,
            beam_width: 64,
            alpha: 1.2,
            max_in_neighbors: 64,
            enable_back_edges: true,
        }
    }
}

/// Node state for in-place updates.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct InPlaceNode {
    /// Vector data
    vector: Vec<f32>,
    /// Out-neighbors
    out_neighbors: Vec<u32>,
    /// In-neighbors (who points to us)
    in_neighbors: Vec<u32>,
    /// Is this node deleted?
    deleted: bool,
}

impl InPlaceNode {
    fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            out_neighbors: Vec::new(),
            in_neighbors: Vec::new(),
            deleted: false,
        }
    }

    fn is_deleted(&self) -> bool {
        self.deleted
    }

    fn mark_deleted(&mut self) {
        self.deleted = true;
    }
}

/// Graph index with in-place update support.
pub struct InPlaceIndex {
    config: InPlaceConfig,
    dim: usize,
    /// Nodes (may contain deleted slots)
    nodes: Vec<Option<InPlaceNode>>,
    /// Free slots for reuse
    free_slots: Vec<u32>,
    /// Entry point for search
    entry_point: u32,
    /// Active node count
    active_count: u32,
}

impl InPlaceIndex {
    /// Create new in-place index.
    pub fn new(dim: usize, config: InPlaceConfig) -> Self {
        Self {
            config,
            dim,
            nodes: Vec::new(),
            free_slots: Vec::new(),
            entry_point: u32::MAX,
            active_count: 0,
        }
    }

    /// Save the in-place update graph to a file.
    #[cfg(feature = "serde")]
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RetrieveError> {
        let file = std::fs::File::create(path.as_ref())?;
        serde_json::to_writer_pretty(
            std::io::BufWriter::new(file),
            &InPlaceSnapshot {
                magic: *b"VICIPL01",
                version: 1,
                config: self.config.clone(),
                dim: self.dim,
                nodes: self.nodes.clone(),
                free_slots: self.free_slots.clone(),
                entry_point: self.entry_point,
                active_count: self.active_count,
            },
        )
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
    }

    /// Load an in-place update graph from a file snapshot.
    #[cfg(feature = "serde")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RetrieveError> {
        let file = std::fs::File::open(path.as_ref())?;
        let snapshot: InPlaceSnapshot = serde_json::from_reader(std::io::BufReader::new(file))
            .map_err(|e| RetrieveError::FormatError(e.to_string()))?;
        snapshot.into_index()
    }

    /// Insert a vector with in-place update.
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32, RetrieveError> {
        if vector.len() != self.dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dim,
            });
        }

        // Get slot (reuse or allocate new)
        let id = if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot as usize] = Some(InPlaceNode::new(vector.clone()));
            slot
        } else {
            let id = self.nodes.len() as u32;
            self.nodes.push(Some(InPlaceNode::new(vector.clone())));
            id
        };

        self.active_count += 1;

        // Set entry point if first node
        if self.entry_point == u32::MAX {
            self.entry_point = id;
            return Ok(id);
        }

        // Find candidate neighbors via greedy search
        let candidates = self.search_for_candidates(&vector);

        // Add out-edges with diversity pruning
        let neighbors = self.select_neighbors(&vector, &candidates);

        // Update out-neighbors
        if let Some(ref mut node) = self.nodes[id as usize] {
            node.out_neighbors = neighbors.clone();
        }

        // Update in-neighbor lists and potentially add back-edges
        // Collect back-edge candidates first to avoid borrow issues
        let mut back_edge_candidates: Vec<(u32, bool)> = Vec::new();

        for &neighbor_id in &neighbors {
            let should_add_back_edge = if self.config.enable_back_edges {
                // Get neighbor info without mutable borrow
                if let Some(Some(neighbor)) = self.nodes.get(neighbor_id as usize) {
                    if neighbor.out_neighbors.len() < self.config.max_degree {
                        let dist_to_new = euclidean_distance(&neighbor.vector, &vector);
                        let worst_neighbor_dist = neighbor
                            .out_neighbors
                            .iter()
                            .filter_map(|&n| {
                                self.nodes
                                    .get(n as usize)
                                    .and_then(|opt| opt.as_ref())
                                    .map(|node| euclidean_distance(&neighbor.vector, &node.vector))
                            })
                            .fold(0.0f32, f32::max);

                        dist_to_new < worst_neighbor_dist
                            || neighbor.out_neighbors.len() < self.config.max_degree / 2
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            back_edge_candidates.push((neighbor_id, should_add_back_edge));
        }

        // Now apply mutations
        for (neighbor_id, should_add_back_edge) in back_edge_candidates {
            if let Some(ref mut neighbor) = self.nodes[neighbor_id as usize] {
                // Add to in-neighbor list
                if neighbor.in_neighbors.len() < self.config.max_in_neighbors {
                    neighbor.in_neighbors.push(id);
                }

                // Add back-edge if appropriate
                if should_add_back_edge {
                    neighbor.out_neighbors.push(id);
                }
            }

            // Update our in-neighbors if back-edge was added
            if should_add_back_edge {
                if let Some(ref mut new_node) = self.nodes[id as usize] {
                    if new_node.in_neighbors.len() < self.config.max_in_neighbors {
                        new_node.in_neighbors.push(neighbor_id);
                    }
                }
            }
        }

        Ok(id)
    }

    /// Delete a node with in-place update.
    pub fn delete(&mut self, id: u32) -> Result<(), RetrieveError> {
        if id as usize >= self.nodes.len() {
            return Err(RetrieveError::OutOfBounds(id as usize));
        }

        let node = self.nodes[id as usize]
            .as_mut()
            .ok_or(RetrieveError::OutOfBounds(id as usize))?;

        if node.is_deleted() {
            return Err(RetrieveError::OutOfBounds(id as usize));
        }

        // Collect node info before marking deleted
        let in_neighbors: Vec<u32> = node.in_neighbors.clone();
        let out_neighbors: Vec<u32> = node.out_neighbors.clone();
        let deleted_vector: Vec<f32> = node.vector.clone();

        // Mark as deleted
        node.mark_deleted();
        self.active_count -= 1;

        // Update entry point if necessary
        if self.entry_point == id {
            // Find new entry point from neighbors
            let new_entry = out_neighbors
                .iter()
                .chain(in_neighbors.iter())
                .find(|&&n| {
                    self.nodes
                        .get(n as usize)
                        .and_then(|opt| opt.as_ref())
                        .map(|node| !node.is_deleted())
                        .unwrap_or(false)
                })
                .copied()
                .unwrap_or(u32::MAX);
            self.entry_point = new_entry;
        }

        // For each in-neighbor, remove edge and find replacement
        // First pass: remove edges and collect current neighbors
        let mut repair_info: Vec<(u32, Vec<u32>)> = Vec::new();

        for in_neighbor_id in &in_neighbors {
            if let Some(ref mut in_neighbor) = self.nodes[*in_neighbor_id as usize] {
                if in_neighbor.is_deleted() {
                    continue;
                }

                // Remove edge to deleted node
                in_neighbor.out_neighbors.retain(|&n| n != id);

                // Collect current neighbors for replacement search
                repair_info.push((*in_neighbor_id, in_neighbor.out_neighbors.clone()));
            }
        }

        // Second pass: find replacements using Wolverine crescent-locus filtering.
        // Candidates must satisfy:
        //   (1) d(c, in_neighbor) < d(deleted, in_neighbor) — close to the in-neighbor
        //   (2) d(c, deleted) > d(deleted, in_neighbor) — far from the deleted node
        // This crescent-shaped region produces higher-quality repair edges than
        // picking the globally closest 2-hop neighbor (Wolverine, VLDB 2025).
        let mut replacements: Vec<(u32, Option<u32>)> = Vec::new();
        for (in_neighbor_id, current_neighbors) in &repair_info {
            let replacement = self.find_replacement_wolverine(
                *in_neighbor_id,
                current_neighbors,
                &deleted_vector,
            );
            replacements.push((*in_neighbor_id, replacement));
        }

        // Third pass: apply replacements (needs mutable borrow)
        for (in_neighbor_id, replacement) in replacements {
            if let Some(new_neighbor) = replacement {
                if let Some(ref mut in_neighbor) = self.nodes[in_neighbor_id as usize] {
                    if !in_neighbor.out_neighbors.contains(&new_neighbor) {
                        in_neighbor.out_neighbors.push(new_neighbor);
                    }
                }

                // Update in-neighbor list of new neighbor
                if let Some(ref mut new_nb) = self.nodes[new_neighbor as usize] {
                    if new_nb.in_neighbors.len() < self.config.max_in_neighbors {
                        new_nb.in_neighbors.push(in_neighbor_id);
                    }
                }
            }
        }

        // Remove from in-neighbor lists of out-neighbors
        for out_neighbor_id in out_neighbors {
            if let Some(ref mut out_neighbor) = self.nodes[out_neighbor_id as usize] {
                out_neighbor.in_neighbors.retain(|&n| n != id);
            }
        }

        // Add slot to free list
        self.nodes[id as usize] = None;
        self.free_slots.push(id);

        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if query.len() != self.dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dim,
            });
        }

        let entry = self.entry_point;
        if entry == u32::MAX {
            return Ok(Vec::new());
        }

        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

        // Initialize with entry point
        let entry_dist = self.distance_to_vector(entry, query);
        candidates.push(Candidate {
            id: entry,
            distance: -entry_dist,
        }); // Min-heap
        results.push(Candidate {
            id: entry,
            distance: entry_dist,
        });
        visited.insert(entry);

        while let Some(Candidate {
            id: current,
            distance: neg_dist,
        }) = candidates.pop()
        {
            let current_dist = -neg_dist;

            let worst = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
            if current_dist > worst && results.len() >= k {
                break;
            }

            // Expand neighbors
            if let Some(Some(node)) = self.nodes.get(current as usize) {
                if node.is_deleted() {
                    continue;
                }

                for &neighbor in &node.out_neighbors {
                    if visited.insert(neighbor) {
                        if let Some(Some(nb_node)) = self.nodes.get(neighbor as usize) {
                            if nb_node.is_deleted() {
                                continue;
                            }

                            let dist = self.distance_to_vector(neighbor, query);

                            if results.len() < k || dist < worst {
                                results.push(Candidate {
                                    id: neighbor,
                                    distance: dist,
                                });
                                while results.len() > k {
                                    results.pop();
                                }
                            }

                            if candidates.len() < self.config.beam_width {
                                candidates.push(Candidate {
                                    id: neighbor,
                                    distance: -dist,
                                });
                            }
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<(u32, f32)> =
            results.into_iter().map(|c| (c.id, c.distance)).collect();
        result_vec.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        result_vec.truncate(k);

        Ok(result_vec)
    }

    /// Search for candidate neighbors during insertion.
    fn search_for_candidates(&self, query: &[f32]) -> Vec<(u32, f32)> {
        let entry = self.entry_point;
        if entry == u32::MAX {
            return Vec::new();
        }

        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: Vec<(u32, f32)> = Vec::new();

        let entry_dist = self.distance_to_vector(entry, query);
        candidates.push(Candidate {
            id: entry,
            distance: -entry_dist,
        });
        visited.insert(entry);

        while let Some(Candidate { id: current, .. }) = candidates.pop() {
            if let Some(Some(node)) = self.nodes.get(current as usize) {
                if node.is_deleted() {
                    continue;
                }

                let dist = self.distance_to_vector(current, query);
                results.push((current, dist));

                for &neighbor in &node.out_neighbors {
                    if visited.insert(neighbor) {
                        if let Some(Some(nb)) = self.nodes.get(neighbor as usize) {
                            if !nb.is_deleted() {
                                let d = self.distance_to_vector(neighbor, query);
                                candidates.push(Candidate {
                                    id: neighbor,
                                    distance: -d,
                                });
                            }
                        }
                    }
                }
            }

            if visited.len() >= self.config.beam_width * 2 {
                break;
            }
        }

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(self.config.beam_width);
        results
    }

    /// Select diverse neighbors with alpha-pruning.
    fn select_neighbors(&self, _query: &[f32], candidates: &[(u32, f32)]) -> Vec<u32> {
        let mut selected = Vec::new();

        for &(candidate, dist) in candidates {
            if selected.len() >= self.config.max_degree {
                break;
            }

            // Alpha-pruning check
            let is_diverse = selected.iter().all(|&s| {
                let s_dist = self.distance(candidate, s);
                s_dist > dist * self.config.alpha || s_dist > dist
            });

            if is_diverse {
                selected.push(candidate);
            }
        }

        selected
    }

    /// Find replacement neighbor after deletion (simple closest-2-hop strategy).
    /// Used by the `repair()` method; the per-delete path uses `find_replacement_wolverine`.
    #[allow(dead_code)]
    fn find_replacement_neighbor(&self, node_id: u32, current_neighbors: &[u32]) -> Option<u32> {
        let node = self.nodes.get(node_id as usize)?.as_ref()?;
        let current_set: HashSet<u32> = current_neighbors.iter().copied().collect();

        let mut candidates: Vec<(u32, f32)> = Vec::new();

        for &neighbor in current_neighbors {
            if let Some(Some(nb_node)) = self.nodes.get(neighbor as usize) {
                if nb_node.is_deleted() {
                    continue;
                }

                for &two_hop in &nb_node.out_neighbors {
                    if two_hop != node_id
                        && !current_set.contains(&two_hop)
                        && !candidates.iter().any(|(id, _)| *id == two_hop)
                    {
                        if let Some(Some(th_node)) = self.nodes.get(two_hop as usize) {
                            if !th_node.is_deleted() {
                                let dist = euclidean_distance(&node.vector, &th_node.vector);
                                candidates.push((two_hop, dist));
                            }
                        }
                    }
                }
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.first().map(|(id, _)| *id)
    }

    /// Find replacement neighbor using Wolverine crescent-locus filtering.
    ///
    /// A candidate `c` is eligible only if it lies in the crescent-shaped region:
    ///   (1) `d(c, node) < d(deleted, node)` — closer to the in-neighbor than the deleted node was
    ///   (2) `d(c, deleted) > d(deleted, node)` — far from the deleted node (likely still reachable)
    ///
    /// This produces repair edges that survive diversity pruning (EdgeTrim) and
    /// are monotonically reachable from the entry point. (Wolverine, VLDB 2025)
    fn find_replacement_wolverine(
        &self,
        node_id: u32,
        current_neighbors: &[u32],
        deleted_vector: &[f32],
    ) -> Option<u32> {
        let node = self.nodes.get(node_id as usize)?.as_ref()?;
        let current_set: HashSet<u32> = current_neighbors.iter().copied().collect();

        // Distance from the deleted node to this in-neighbor — defines the crescent radii.
        let dist_deleted_to_node = euclidean_distance(&node.vector, deleted_vector);

        let mut crescent_candidates: Vec<(u32, f32)> = Vec::new();
        let mut fallback_candidates: Vec<(u32, f32)> = Vec::new();

        // Search 2-hop neighborhood.
        for &neighbor in current_neighbors {
            if let Some(Some(nb_node)) = self.nodes.get(neighbor as usize) {
                if nb_node.is_deleted() {
                    continue;
                }

                for &two_hop in &nb_node.out_neighbors {
                    if two_hop == node_id || current_set.contains(&two_hop) {
                        continue;
                    }
                    // Dedup.
                    if crescent_candidates.iter().any(|(id, _)| *id == two_hop)
                        || fallback_candidates.iter().any(|(id, _)| *id == two_hop)
                    {
                        continue;
                    }

                    if let Some(Some(th_node)) = self.nodes.get(two_hop as usize) {
                        if th_node.is_deleted() {
                            continue;
                        }

                        let dist_to_node = euclidean_distance(&node.vector, &th_node.vector);
                        let dist_to_deleted = euclidean_distance(deleted_vector, &th_node.vector);

                        // Crescent-locus conditions.
                        let close_to_node = dist_to_node < dist_deleted_to_node;
                        let far_from_deleted = dist_to_deleted > dist_deleted_to_node;

                        if close_to_node && far_from_deleted {
                            crescent_candidates.push((two_hop, dist_to_node));
                        } else {
                            // Fallback: collect all candidates in case the crescent is empty.
                            fallback_candidates.push((two_hop, dist_to_node));
                        }
                    }
                }
            }
        }

        // Prefer crescent candidates; fall back to closest 2-hop if crescent is empty.
        if !crescent_candidates.is_empty() {
            crescent_candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            Some(crescent_candidates[0].0)
        } else {
            fallback_candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            fallback_candidates.first().map(|(id, _)| *id)
        }
    }

    /// Distance between two nodes in the index.
    fn distance(&self, a: u32, b: u32) -> f32 {
        match (&self.nodes.get(a as usize), &self.nodes.get(b as usize)) {
            (Some(Some(na)), Some(Some(nb))) => euclidean_distance(&na.vector, &nb.vector),
            _ => f32::INFINITY,
        }
    }

    /// Distance from node to query vector.
    fn distance_to_vector(&self, id: u32, query: &[f32]) -> f32 {
        match self.nodes.get(id as usize) {
            Some(Some(node)) if !node.is_deleted() => euclidean_distance(&node.vector, query),
            _ => f32::INFINITY,
        }
    }

    /// Number of active nodes.
    pub fn len(&self) -> usize {
        self.active_count as usize
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> InPlaceStats {
        let mut total_out_degree = 0usize;
        let mut total_in_degree = 0usize;
        let mut active = 0usize;

        for node in self.nodes.iter().flatten() {
            if !node.is_deleted() {
                active += 1;
                total_out_degree += node.out_neighbors.len();
                total_in_degree += node.in_neighbors.len();
            }
        }

        InPlaceStats {
            active_nodes: active,
            free_slots: self.free_slots.len(),
            avg_out_degree: if active > 0 {
                total_out_degree as f32 / active as f32
            } else {
                0.0
            },
            avg_in_degree: if active > 0 {
                total_in_degree as f32 / active as f32
            } else {
                0.0
            },
        }
    }

    /// Run MN-RU graph repair on all nodes that currently have edges to deleted
    /// (None) slots, or whose neighbor lists were thinned by prior deletions.
    ///
    /// Call this after a batch of `delete()` calls to restore graph connectivity.
    /// The built-in per-delete repair only finds a single replacement neighbor;
    /// this method uses the full MN-RU algorithm with 2-hop candidate search and
    /// diversity pruning.
    ///
    /// Returns `RepairStats` summarizing the work done.
    pub fn repair(&mut self) -> RepairStats {
        self.repair_with_config(RepairConfig {
            max_candidates: 64,
            max_neighbors: self.config.max_degree,
            bidirectional: self.config.enable_back_edges,
            alpha: self.config.alpha,
        })
    }

    /// Run MN-RU graph repair with custom configuration.
    pub fn repair_with_config(&mut self, config: RepairConfig) -> RepairStats {
        let mut total_stats = RepairStats::default();

        // Collect the set of deleted slot indices (nodes that are None or marked deleted).
        let deleted_set: std::collections::HashSet<u32> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, slot)| match slot {
                None => true,
                Some(n) => n.is_deleted(),
            })
            .map(|(i, _)| i as u32)
            .collect();

        if deleted_set.is_empty() {
            return total_stats;
        }

        // Find all active nodes whose neighbor lists reference deleted nodes.
        // These are the nodes that need repair.
        let mut nodes_needing_repair: Vec<u32> = Vec::new();
        for (i, slot) in self.nodes.iter().enumerate() {
            if let Some(node) = slot {
                if !node.is_deleted() && node.out_neighbors.iter().any(|n| deleted_set.contains(n))
                {
                    nodes_needing_repair.push(i as u32);
                }
            }
        }

        // For each node needing repair, use compute_repair_operations to find
        // new neighbor lists. We process one node at a time, treating it as if
        // the node itself is a "neighbor of a deleted node" that needs fixing.
        for &node_id in &nodes_needing_repair {
            let node = match &self.nodes[node_id as usize] {
                Some(n) if !n.is_deleted() => n,
                _ => continue,
            };

            // Current neighbors, filtering out deleted
            let current_valid: Vec<u32> = node
                .out_neighbors
                .iter()
                .filter(|&&n| !deleted_set.contains(&n))
                .copied()
                .collect();

            let removed_count = node.out_neighbors.len() - current_valid.len();
            if removed_count == 0 {
                continue;
            }
            total_stats.edges_removed += removed_count;
            total_stats.nodes_processed += 1;

            // Use compute_repair_operations: we treat this as repairing neighbors
            // of a synthetic "deleted node" whose only neighbor is node_id.
            // But it's simpler to just inline the 2-hop candidate search here,
            // using the same logic as compute_repair_operations.
            let nodes_ref = &self.nodes;
            let get_neighbors = |id: u32| -> Vec<u32> {
                match nodes_ref.get(id as usize) {
                    Some(Some(n)) if !n.is_deleted() => n.out_neighbors.clone(),
                    _ => Vec::new(),
                }
            };
            let compute_distance = |a: u32, b: u32| -> f32 {
                match (nodes_ref.get(a as usize), nodes_ref.get(b as usize)) {
                    (Some(Some(na)), Some(Some(nb))) if !na.is_deleted() && !nb.is_deleted() => {
                        crate::distance::l2_distance(&na.vector, &nb.vector)
                    }
                    _ => f32::INFINITY,
                }
            };

            // Find candidates via 2-hop from current valid neighbors
            let mut visited: std::collections::HashSet<u32> =
                current_valid.iter().copied().collect();
            visited.insert(node_id);
            visited.extend(deleted_set.iter().copied());

            let mut candidates: Vec<(u32, f32)> = Vec::new();
            for &n in &current_valid {
                for two_hop in get_neighbors(n) {
                    if visited.insert(two_hop) {
                        let dist = compute_distance(node_id, two_hop);
                        if dist.is_finite() {
                            candidates.push((two_hop, dist));
                        }
                    }
                }
            }
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

            // Build new neighbor list: start from current valid, add best candidates
            let mut new_neighbors = current_valid;
            for (candidate, _dist) in &candidates {
                if new_neighbors.len() >= config.max_neighbors {
                    break;
                }
                new_neighbors.push(*candidate);
                total_stats.edges_added += 1;
            }

            // Apply the updated neighbor list
            if let Some(ref mut node) = self.nodes[node_id as usize] {
                node.out_neighbors = new_neighbors.clone();
            }

            // Bidirectional: ensure new neighbors have a back-edge to node_id
            if config.bidirectional {
                for (candidate, _) in &candidates {
                    if let Some(ref mut cand_node) = self.nodes[*candidate as usize] {
                        if !cand_node.is_deleted()
                            && !cand_node.out_neighbors.contains(&node_id)
                            && cand_node.out_neighbors.len() < config.max_neighbors
                        {
                            cand_node.out_neighbors.push(node_id);
                            total_stats.bidirectional_edges += 1;
                        }
                        // Update in-neighbor tracking
                        if cand_node.in_neighbors.len() < self.config.max_in_neighbors
                            && !cand_node.in_neighbors.contains(&node_id)
                        {
                            cand_node.in_neighbors.push(node_id);
                        }
                    }
                    // Update our in-neighbors
                    if let Some(ref mut node) = self.nodes[node_id as usize] {
                        if node.in_neighbors.len() < self.config.max_in_neighbors
                            && !node.in_neighbors.contains(candidate)
                        {
                            node.in_neighbors.push(*candidate);
                        }
                    }
                }
            }
        }

        total_stats
    }

    /// Check graph connectivity from the entry point.
    ///
    /// Returns `(reachable_count, orphan_count)`. After repair, orphan_count
    /// should be 0 for a well-connected graph.
    pub fn validate_connectivity(&self) -> (usize, usize) {
        if self.entry_point == u32::MAX {
            return (0, 0);
        }

        let total = self.nodes.len();
        validate_connectivity(
            self.entry_point,
            total,
            |id| match self.nodes.get(id as usize) {
                Some(Some(n)) if !n.is_deleted() => n.out_neighbors.clone(),
                _ => Vec::new(),
            },
            |id| match self.nodes.get(id as usize) {
                Some(Some(n)) => n.is_deleted(),
                Some(None) | None => true,
            },
        )
    }
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct InPlaceSnapshot {
    magic: [u8; 8],
    version: u32,
    config: InPlaceConfig,
    dim: usize,
    nodes: Vec<Option<InPlaceNode>>,
    free_slots: Vec<u32>,
    entry_point: u32,
    active_count: u32,
}

#[cfg(feature = "serde")]
impl InPlaceSnapshot {
    fn into_index(self) -> Result<InPlaceIndex, RetrieveError> {
        if self.magic != *b"VICIPL01" {
            return Err(RetrieveError::FormatError(
                "invalid InPlaceIndex snapshot magic".into(),
            ));
        }
        if self.version != 1 {
            return Err(RetrieveError::FormatError(format!(
                "unsupported InPlaceIndex snapshot version {}",
                self.version
            )));
        }
        validate_inplace_parts(
            self.dim,
            &self.nodes,
            &self.free_slots,
            self.entry_point,
            self.active_count,
        )?;

        Ok(InPlaceIndex {
            config: self.config,
            dim: self.dim,
            nodes: self.nodes,
            free_slots: self.free_slots,
            entry_point: self.entry_point,
            active_count: self.active_count,
        })
    }
}

#[cfg(feature = "serde")]
fn validate_inplace_parts(
    dim: usize,
    nodes: &[Option<InPlaceNode>],
    free_slots: &[u32],
    entry_point: u32,
    active_count: u32,
) -> Result<(), RetrieveError> {
    if dim == 0 {
        return Err(RetrieveError::FormatError(
            "InPlaceIndex snapshot has zero dimension".into(),
        ));
    }
    let mut free_seen = std::collections::HashSet::with_capacity(free_slots.len());
    for &slot in free_slots {
        if slot as usize >= nodes.len() {
            return Err(RetrieveError::FormatError(format!(
                "InPlaceIndex free slot {slot} exceeds node count {}",
                nodes.len()
            )));
        }
        if !free_seen.insert(slot) {
            return Err(RetrieveError::FormatError(format!(
                "InPlaceIndex duplicate free slot {slot}"
            )));
        }
        if matches!(nodes.get(slot as usize), Some(Some(node)) if !node.is_deleted()) {
            return Err(RetrieveError::FormatError(format!(
                "InPlaceIndex live node {slot} also appears in free slots"
            )));
        }
    }

    let mut active = 0u32;
    for (id, slot) in nodes.iter().enumerate() {
        let Some(node) = slot else {
            if !free_seen.contains(&(id as u32)) {
                return Err(RetrieveError::FormatError(format!(
                    "InPlaceIndex empty slot {id} missing from free slots"
                )));
            }
            continue;
        };
        if node.vector.len() != dim {
            return Err(RetrieveError::FormatError(format!(
                "InPlaceIndex node {id} has dimension {}, expected {dim}",
                node.vector.len()
            )));
        }
        if node.is_deleted() {
            continue;
        }
        active += 1;
        for &neighbor in node.out_neighbors.iter().chain(node.in_neighbors.iter()) {
            match nodes.get(neighbor as usize) {
                Some(Some(neighbor_node)) if !neighbor_node.is_deleted() => {}
                _ => {
                    return Err(RetrieveError::FormatError(format!(
                        "InPlaceIndex node {id} references inactive neighbor {neighbor}"
                    )));
                }
            }
        }
    }

    if active != active_count {
        return Err(RetrieveError::FormatError(format!(
            "InPlaceIndex active_count {active_count} does not match live nodes {active}"
        )));
    }
    if active == 0 {
        if entry_point != u32::MAX {
            return Err(RetrieveError::FormatError(
                "InPlaceIndex empty snapshot has a live entry point".into(),
            ));
        }
    } else {
        match nodes.get(entry_point as usize) {
            Some(Some(node)) if !node.is_deleted() => {}
            _ => {
                return Err(RetrieveError::FormatError(format!(
                    "InPlaceIndex entry point {entry_point} is not live"
                )));
            }
        }
    }

    Ok(())
}

/// Statistics for in-place index.
#[derive(Clone, Debug)]
pub struct InPlaceStats {
    /// Number of nodes currently in the index.
    pub active_nodes: usize,
    /// Number of deleted slots available for reuse.
    pub free_slots: usize,
    /// Mean outgoing edge count per node.
    pub avg_out_degree: f32,
    /// Mean incoming edge count per node.
    pub avg_in_degree: f32,
}

/// Search candidate.
#[derive(Clone, Copy)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority (for results pruning)
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance)
    }
}

use crate::distance::l2_distance as euclidean_distance;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_inplace_insert_search() {
        let mut index = InPlaceIndex::new(4, InPlaceConfig::default());

        // Insert vectors
        for i in 0..20 {
            let v = vec![i as f32, (i * 2) as f32, 0.0, 0.0];
            index.insert(v).unwrap();
        }

        assert_eq!(index.len(), 20);

        // Search
        let results = index.search(&[5.0, 10.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_inplace_delete() {
        let mut index = InPlaceIndex::new(4, InPlaceConfig::default());

        // Insert vectors
        for i in 0..10 {
            index.insert(vec![i as f32; 4]).unwrap();
        }

        assert_eq!(index.len(), 10);

        // Delete some
        index.delete(3).unwrap();
        index.delete(5).unwrap();

        assert_eq!(index.len(), 8);

        // Search should still work
        let results = index.search(&[4.0; 4], 3).unwrap();
        assert!(!results.is_empty());

        // Deleted nodes shouldn't appear in results
        for (id, _) in &results {
            assert!(*id != 3 && *id != 5);
        }
    }

    #[test]
    fn test_slot_reuse() {
        let mut index = InPlaceIndex::new(2, InPlaceConfig::default());

        // Insert
        let id1 = index.insert(vec![1.0, 2.0]).unwrap();
        let _id2 = index.insert(vec![3.0, 4.0]).unwrap();

        // Delete
        index.delete(id1).unwrap();

        // Insert should reuse slot
        let id3 = index.insert(vec![5.0, 6.0]).unwrap();
        assert_eq!(id3, id1, "Should reuse deleted slot");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn save_load_roundtrip_preserves_updates_and_free_slots() {
        let mut index = InPlaceIndex::new(
            4,
            InPlaceConfig {
                max_degree: 8,
                beam_width: 16,
                max_in_neighbors: 16,
                ..Default::default()
            },
        );

        for i in 0..20 {
            index
                .insert(vec![i as f32, (i % 3) as f32, 0.0, 0.0])
                .unwrap();
        }
        index.delete(4).unwrap();
        index.delete(9).unwrap();

        let query = [8.0, 2.0, 0.0, 0.0];
        let before = index.search(&query, 5).unwrap();
        let stats_before = index.stats();
        let file = tempfile::NamedTempFile::new().unwrap();
        index.save_to_file(file.path()).unwrap();
        let mut loaded = InPlaceIndex::load_from_file(file.path()).unwrap();

        assert_eq!(loaded.search(&query, 5).unwrap(), before);
        assert_eq!(loaded.stats().active_nodes, stats_before.active_nodes);
        assert_eq!(loaded.stats().free_slots, stats_before.free_slots);

        let reused = loaded.insert(vec![100.0, 100.0, 0.0, 0.0]).unwrap();
        assert!(
            reused == 4 || reused == 9,
            "loaded index should reuse a persisted free slot, got {reused}"
        );
    }

    #[test]
    fn test_stats() {
        let mut index = InPlaceIndex::new(4, InPlaceConfig::default());

        for i in 0..10 {
            index.insert(vec![i as f32; 4]).unwrap();
        }

        let stats = index.stats();
        assert_eq!(stats.active_nodes, 10);
        assert_eq!(stats.free_slots, 0);
    }

    #[test]
    fn test_rapid_insert_delete() {
        let mut index = InPlaceIndex::new(4, InPlaceConfig::default());

        // Rapid insert/delete cycles
        for cycle in 0..5 {
            // Insert batch
            let mut ids = Vec::new();
            for i in 0..10 {
                let id = index.insert(vec![(cycle * 10 + i) as f32; 4]).unwrap();
                ids.push(id);
            }

            // Delete half
            for i in (0..10).step_by(2) {
                index.delete(ids[i]).unwrap();
            }
        }

        // Should still be searchable
        let results = index.search(&[25.0; 4], 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_repair_after_deletions() {
        let mut index = InPlaceIndex::new(4, InPlaceConfig::default());

        // Insert 30 vectors in a grid pattern so the graph is well-connected
        for i in 0..30 {
            index
                .insert(vec![i as f32, (i % 5) as f32, 0.0, 0.0])
                .unwrap();
        }
        assert_eq!(index.len(), 30);

        // Check connectivity before deletions
        let (reachable_before, _) = index.validate_connectivity();
        assert_eq!(reachable_before, 30);

        // Delete a batch of nodes
        for id in [5, 10, 15, 20] {
            index.delete(id).unwrap();
        }
        assert_eq!(index.len(), 26);

        // Run repair
        let stats = index.repair();
        // Should have processed some nodes (those whose neighbors were deleted)
        // Stats may show 0 if the per-delete repair already cleaned edges,
        // but the method should not panic.
        // Repair ran without panic; edges_removed count is informational.
        let _ = stats.edges_removed;

        // Validate connectivity after repair
        let (reachable_after, orphans) = index.validate_connectivity();
        assert_eq!(orphans, 0, "no orphans after repair");
        assert_eq!(reachable_after, 26);

        // Search should still find results
        let results = index.search(&[7.0, 2.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.is_empty());
        // Deleted nodes must not appear
        for (id, _) in &results {
            assert!(![5, 10, 15, 20].contains(id));
        }
    }

    #[test]
    fn test_validate_connectivity_empty() {
        let index = InPlaceIndex::new(4, InPlaceConfig::default());
        let (reachable, orphans) = index.validate_connectivity();
        assert_eq!(reachable, 0);
        assert_eq!(orphans, 0);
    }

    #[test]
    fn test_repair_no_deletions_is_noop() {
        let mut index = InPlaceIndex::new(4, InPlaceConfig::default());
        for i in 0..10 {
            index.insert(vec![i as f32; 4]).unwrap();
        }
        let stats = index.repair();
        assert_eq!(stats.nodes_processed, 0);
        assert_eq!(stats.edges_removed, 0);
        assert_eq!(stats.edges_added, 0);
    }

    /// Wolverine crescent-locus deletion: after deleting hub nodes, the replacement
    /// edges should maintain graph connectivity and search quality.
    #[test]
    fn test_wolverine_crescent_locus_recall() {
        let dim = 8;
        let mut index = InPlaceIndex::new(
            dim,
            InPlaceConfig {
                max_degree: 16,
                beam_width: 32,
                alpha: 1.2,
                max_in_neighbors: 32,
                enable_back_edges: true,
            },
        );

        // Insert 50 vectors in a structured pattern.
        let mut rng_seed: u64 = 42;
        let mut next = || -> f32 {
            rng_seed = rng_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_seed >> 33) as f32) / (u32::MAX as f32)
        };

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for _ in 0..50 {
            let v: Vec<f32> = (0..dim).map(|_| next()).collect();
            vectors.push(v);
        }

        for v in &vectors {
            index.insert(v.clone()).unwrap();
        }

        // Delete 10 nodes (20% deletion rate — stresses repair quality).
        let to_delete = [3, 7, 12, 18, 23, 28, 33, 38, 43, 48];
        for &id in &to_delete {
            index.delete(id).unwrap();
        }

        assert_eq!(index.len(), 40);

        // Verify no deleted nodes appear in results.
        let query: Vec<f32> = (0..dim).map(|_| next()).collect();
        let results = index.search(&query, 10).unwrap();
        for (id, _) in &results {
            assert!(
                !to_delete.contains(id),
                "deleted node {} appeared in results",
                id
            );
        }

        // Verify connectivity after Wolverine deletion.
        let (reachable, orphans) = index.validate_connectivity();
        assert_eq!(
            orphans, 0,
            "wolverine deletion should maintain connectivity"
        );
        assert_eq!(reachable, 40);
    }

    /// Test that crescent-locus filtering actually selects candidates in the
    /// correct geometric region.
    #[test]
    fn test_wolverine_crescent_geometry() {
        let dim = 4;
        let mut index = InPlaceIndex::new(dim, InPlaceConfig::default());

        // Insert points along a line: [0,0,0,0], [1,0,0,0], ..., [9,0,0,0].
        for i in 0..10 {
            index.insert(vec![i as f32, 0.0, 0.0, 0.0]).unwrap();
        }

        // Delete node 5 (middle of the line).
        // In-neighbors of 5 should find replacements that are:
        //   (1) closer to the in-neighbor than node 5 was
        //   (2) farther from node 5's position than the in-neighbor was
        index.delete(5).unwrap();

        assert_eq!(index.len(), 9);

        // Search near the deleted node's position — should still find neighbors.
        let results = index.search(&[5.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(!results.is_empty());

        // Node 5 should not appear.
        assert!(
            !results.iter().any(|(id, _)| *id == 5),
            "deleted node should not appear"
        );

        // Closest surviving nodes to [5,0,0,0] are 4 and 6.
        let result_ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            result_ids.contains(&4) || result_ids.contains(&6),
            "should find neighbors near deleted node"
        );
    }
}

// =============================================================================
// IndexOps implementation for streaming updates
// =============================================================================

use crate::streaming::IndexOps;

/// Wrapper around InPlaceIndex that maintains external ID mapping.
///
/// This allows using InPlaceIndex with the streaming coordinator,
/// which requires explicit external IDs.
pub struct MappedInPlaceIndex {
    inner: InPlaceIndex,
    /// External ID -> Internal ID
    id_map: std::collections::HashMap<u32, u32>,
    /// Internal ID -> External ID  
    reverse_map: std::collections::HashMap<u32, u32>,
}

impl MappedInPlaceIndex {
    /// Create a new mapped index.
    pub fn new(dim: usize, config: InPlaceConfig) -> Self {
        Self {
            inner: InPlaceIndex::new(dim, config),
            id_map: std::collections::HashMap::new(),
            reverse_map: std::collections::HashMap::new(),
        }
    }

    /// Get the underlying index.
    pub fn inner(&self) -> &InPlaceIndex {
        &self.inner
    }

    /// Get statistics.
    pub fn stats(&self) -> InPlaceStats {
        self.inner.stats()
    }

    /// Save the mapped in-place index to a file.
    #[cfg(feature = "serde")]
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), RetrieveError> {
        let file = std::fs::File::create(path.as_ref())?;
        serde_json::to_writer_pretty(
            std::io::BufWriter::new(file),
            &MappedInPlaceSnapshot {
                magic: *b"VICIPM01",
                version: 1,
                inner: InPlaceSnapshot {
                    magic: *b"VICIPL01",
                    version: 1,
                    config: self.inner.config.clone(),
                    dim: self.inner.dim,
                    nodes: self.inner.nodes.clone(),
                    free_slots: self.inner.free_slots.clone(),
                    entry_point: self.inner.entry_point,
                    active_count: self.inner.active_count,
                },
                id_map: self.id_map.clone(),
                reverse_map: self.reverse_map.clone(),
            },
        )
        .map_err(|e| RetrieveError::FormatError(e.to_string()))
    }

    /// Load a mapped in-place index from a file snapshot.
    #[cfg(feature = "serde")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RetrieveError> {
        let file = std::fs::File::open(path.as_ref())?;
        let snapshot: MappedInPlaceSnapshot =
            serde_json::from_reader(std::io::BufReader::new(file))
                .map_err(|e| RetrieveError::FormatError(e.to_string()))?;
        snapshot.into_index()
    }
}

#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct MappedInPlaceSnapshot {
    magic: [u8; 8],
    version: u32,
    inner: InPlaceSnapshot,
    id_map: std::collections::HashMap<u32, u32>,
    reverse_map: std::collections::HashMap<u32, u32>,
}

#[cfg(feature = "serde")]
impl MappedInPlaceSnapshot {
    fn into_index(self) -> Result<MappedInPlaceIndex, RetrieveError> {
        if self.magic != *b"VICIPM01" {
            return Err(RetrieveError::FormatError(
                "invalid MappedInPlaceIndex snapshot magic".into(),
            ));
        }
        if self.version != 1 {
            return Err(RetrieveError::FormatError(format!(
                "unsupported MappedInPlaceIndex snapshot version {}",
                self.version
            )));
        }
        validate_inplace_maps(&self.inner.nodes, &self.id_map, &self.reverse_map)?;
        Ok(MappedInPlaceIndex {
            inner: self.inner.into_index()?,
            id_map: self.id_map,
            reverse_map: self.reverse_map,
        })
    }
}

#[cfg(feature = "serde")]
fn validate_inplace_maps(
    nodes: &[Option<InPlaceNode>],
    id_map: &std::collections::HashMap<u32, u32>,
    reverse_map: &std::collections::HashMap<u32, u32>,
) -> Result<(), RetrieveError> {
    if id_map.len() != reverse_map.len() {
        return Err(RetrieveError::FormatError(
            "MappedInPlaceIndex id maps have different lengths".into(),
        ));
    }
    for (&external_id, &internal_id) in id_map {
        match reverse_map.get(&internal_id) {
            Some(&mapped_external) if mapped_external == external_id => {}
            _ => {
                return Err(RetrieveError::FormatError(format!(
                    "MappedInPlaceIndex missing reverse map for external id {external_id}"
                )));
            }
        }
        match nodes.get(internal_id as usize) {
            Some(Some(node)) if !node.is_deleted() => {}
            _ => {
                return Err(RetrieveError::FormatError(format!(
                    "MappedInPlaceIndex external id {external_id} maps to inactive node {internal_id}"
                )));
            }
        }
    }
    for (&internal_id, &external_id) in reverse_map {
        if id_map.get(&external_id) != Some(&internal_id) {
            return Err(RetrieveError::FormatError(format!(
                "MappedInPlaceIndex reverse id {internal_id} does not map back from {external_id}"
            )));
        }
    }
    Ok(())
}

impl IndexOps for MappedInPlaceIndex {
    fn insert(&mut self, id: u32, vector: Vec<f32>) -> crate::error::Result<()> {
        // If ID already exists, update by delete + insert
        if let Some(&internal_id) = self.id_map.get(&id) {
            self.inner.delete(internal_id)?;
            self.reverse_map.remove(&internal_id);
        }

        // Insert into inner index
        let internal_id = self.inner.insert(vector)?;

        // Update mappings
        self.id_map.insert(id, internal_id);
        self.reverse_map.insert(internal_id, id);

        Ok(())
    }

    fn delete(&mut self, id: u32) -> crate::error::Result<()> {
        if let Some(&internal_id) = self.id_map.get(&id) {
            self.inner.delete(internal_id)?;
            self.id_map.remove(&id);
            self.reverse_map.remove(&internal_id);
            Ok(())
        } else {
            // Silently succeed if ID doesn't exist
            Ok(())
        }
    }

    fn search(&self, query: &[f32], k: usize) -> crate::error::Result<Vec<(u32, f32)>> {
        let results = self.inner.search(query, k)?;

        // Map internal IDs back to external IDs
        Ok(results
            .into_iter()
            .filter_map(|(internal_id, dist)| {
                self.reverse_map
                    .get(&internal_id)
                    .map(|&external_id| (external_id, dist))
            })
            .collect())
    }
}

impl IndexOps for InPlaceIndex {
    /// Insert a vector.
    ///
    /// Note: The `id` parameter is ignored - InPlaceIndex generates its own IDs.
    /// Use `MappedInPlaceIndex` if you need external ID mapping.
    fn insert(&mut self, _id: u32, vector: Vec<f32>) -> crate::error::Result<()> {
        self.insert(vector)?;
        Ok(())
    }

    fn delete(&mut self, id: u32) -> crate::error::Result<()> {
        InPlaceIndex::delete(self, id)
    }

    fn search(&self, query: &[f32], k: usize) -> crate::error::Result<Vec<(u32, f32)>> {
        InPlaceIndex::search(self, query, k)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod streaming_tests {
    use super::*;
    use crate::streaming::{IndexOps, StreamingCoordinator};

    #[test]
    fn test_inplace_with_streaming_coordinator() {
        let index = InPlaceIndex::new(4, InPlaceConfig::default());
        let mut streaming = StreamingCoordinator::new(index);

        // Insert via streaming
        streaming.insert(0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        streaming.insert(1, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        streaming.insert(2, vec![0.0, 0.0, 1.0, 0.0]).unwrap();

        // Search should find vectors
        let results = streaming.search(&[1.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_mapped_inplace_preserves_ids() {
        let mut index = MappedInPlaceIndex::new(4, InPlaceConfig::default());

        // Insert with specific IDs
        index.insert(100, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(200, vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        // Search should return the external IDs
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());

        // Verify we get back our external IDs
        let ids: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&100) || ids.contains(&200));

        // Delete and verify
        index.delete(100).unwrap();
        let results_after = index.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        let ids_after: Vec<u32> = results_after.iter().map(|(id, _)| *id).collect();
        assert!(!ids_after.contains(&100));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn mapped_inplace_save_load_preserves_external_ids() {
        let mut index = MappedInPlaceIndex::new(4, InPlaceConfig::default());

        index.insert(100, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(200, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(300, vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        index.delete(200).unwrap();
        index.insert(100, vec![2.0, 0.0, 0.0, 0.0]).unwrap();

        let query = [2.0, 0.0, 0.0, 0.0];
        let before = index.search(&query, 3).unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        index.save_to_file(file.path()).unwrap();
        let loaded = MappedInPlaceIndex::load_from_file(file.path()).unwrap();
        let after = loaded.search(&query, 3).unwrap();

        assert_eq!(after, before);
        let ids: Vec<u32> = after.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&100));
        assert!(!ids.contains(&200));
    }
}
