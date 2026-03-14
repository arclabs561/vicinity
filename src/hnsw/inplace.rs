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

use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Configuration for in-place updates.
#[derive(Clone, Debug)]
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
struct InPlaceNode {
    /// Vector data
    vector: Vec<f32>,
    /// Out-neighbors
    out_neighbors: Vec<u32>,
    /// In-neighbors (who points to us)
    in_neighbors: Vec<u32>,
    /// Is this node deleted?
    deleted: AtomicBool,
}

impl InPlaceNode {
    fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            out_neighbors: Vec::new(),
            in_neighbors: Vec::new(),
            deleted: AtomicBool::new(false),
        }
    }

    fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::Acquire)
    }

    fn mark_deleted(&self) {
        self.deleted.store(true, Ordering::Release);
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
    entry_point: AtomicU32,
    /// Active node count
    active_count: AtomicU32,
}

impl InPlaceIndex {
    /// Create new in-place index.
    pub fn new(dim: usize, config: InPlaceConfig) -> Self {
        Self {
            config,
            dim,
            nodes: Vec::new(),
            free_slots: Vec::new(),
            entry_point: AtomicU32::new(u32::MAX),
            active_count: AtomicU32::new(0),
        }
    }

    /// Insert a vector with in-place update.
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<u32, RetrieveError> {
        if vector.len() != self.dim {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dim,
                doc_dim: vector.len(),
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

        self.active_count.fetch_add(1, Ordering::Relaxed);

        // Set entry point if first node
        if self.entry_point.load(Ordering::Acquire) == u32::MAX {
            self.entry_point.store(id, Ordering::Release);
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
            .as_ref()
            .ok_or(RetrieveError::OutOfBounds(id as usize))?;

        if node.is_deleted() {
            return Err(RetrieveError::OutOfBounds(id as usize));
        }

        // Collect in-neighbors before marking deleted
        let in_neighbors: Vec<u32> = node.in_neighbors.clone();
        let out_neighbors: Vec<u32> = node.out_neighbors.clone();
        let _deleted_vector = node.vector.clone();

        // Mark as deleted
        node.mark_deleted();
        self.active_count.fetch_sub(1, Ordering::Relaxed);

        // Update entry point if necessary
        if self.entry_point.load(Ordering::Acquire) == id {
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
            self.entry_point.store(new_entry, Ordering::Release);
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

        // Second pass: find replacements (needs immutable borrow)
        let mut replacements: Vec<(u32, Option<u32>)> = Vec::new();
        for (in_neighbor_id, current_neighbors) in &repair_info {
            let replacement = self.find_replacement_neighbor(*in_neighbor_id, current_neighbors);
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
                query_dim: self.dim,
                doc_dim: query.len(),
            });
        }

        let entry = self.entry_point.load(Ordering::Acquire);
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
        result_vec.sort_by(|a, b| a.1.total_cmp(&b.1));
        result_vec.truncate(k);

        Ok(result_vec)
    }

    /// Search for candidate neighbors during insertion.
    fn search_for_candidates(&self, query: &[f32]) -> Vec<(u32, f32)> {
        let entry = self.entry_point.load(Ordering::Acquire);
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

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
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

    /// Find replacement neighbor after deletion.
    fn find_replacement_neighbor(&self, node_id: u32, current_neighbors: &[u32]) -> Option<u32> {
        let node = self.nodes.get(node_id as usize)?.as_ref()?;
        let current_set: HashSet<u32> = current_neighbors.iter().cloned().collect();

        // Search in 2-hop neighborhood for replacement
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

        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        candidates.first().map(|(id, _)| *id)
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
        self.active_count.load(Ordering::Relaxed) as usize
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
}

/// Statistics for in-place index.
#[derive(Clone, Debug)]
pub struct InPlaceStats {
    pub active_nodes: usize,
    pub free_slots: usize,
    pub avg_out_degree: f32,
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

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

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
}
