//! HNSW search algorithm with early termination optimizations.

use std::cell::RefCell;
use std::collections::{BinaryHeap, HashSet};

// ─── Software prefetch ──────────────────────────────────────────────────────

/// Prefetch a memory address into L1 cache for reading.
///
/// No-op on unsupported platforms. This is a performance hint only;
/// correctness does not depend on it.
#[inline(always)]
#[allow(unsafe_code, unused_variables)]
fn prefetch_read_data(ptr: *const f32) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: _mm_prefetch is a hint; invalid addresses are silently ignored.
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: prefetch is a hint; invalid addresses are silently ignored.
        // Using inline asm because the intrinsic is nightly-only.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
        }
    }
}

#[cfg(feature = "hnsw")]
use crate::distance::cosine_distance_normalized as cosine_distance;

// ─── Visited set ─────────────────────────────────────────────────────────────

/// Threshold below which we use a dense generation-counter array instead of HashSet.
/// 100K nodes = 200KB Vec<u16>, fits comfortably in L2 cache.
const DENSE_VISITED_THRESHOLD: usize = 100_000;

/// Fast visited-node tracker using the generation-counter pattern.
///
/// Dense variant: a `Vec<u16>` where `visited[id] == generation` means visited.
/// Incrementing `generation` logically clears the set in O(1). Only when the
/// u16 counter wraps (every 65 535 searches) does a memset occur.
///
/// Falls back to `HashSet<u32>` for large indexes where a full dense array
/// would waste memory.
enum VisitedSet {
    Dense {
        marks: Vec<u16>,
        generation: u16,
    },
    Sparse(HashSet<u32>),
}

impl VisitedSet {
    /// Create a visited set sized for `num_nodes` total nodes.
    fn new(num_nodes: usize, capacity_hint: usize) -> Self {
        if num_nodes <= DENSE_VISITED_THRESHOLD {
            VisitedSet::Dense {
                marks: vec![0u16; num_nodes],
                generation: 1,
            }
        } else {
            VisitedSet::Sparse(HashSet::with_capacity(capacity_hint))
        }
    }

    /// Reset the visited set for a new search. O(1) amortized for the dense
    /// variant (increments generation; memsets only on u16 overflow).
    /// For the sparse variant, clears the HashSet.
    fn clear(&mut self) {
        match self {
            VisitedSet::Dense { marks, generation } => {
                if let Some(next) = generation.checked_add(1) {
                    *generation = next;
                } else {
                    // Overflow: reset all marks and restart at generation 1
                    marks.fill(0);
                    *generation = 1;
                }
            }
            VisitedSet::Sparse(s) => s.clear(),
        }
    }

    /// Mark a node as visited. Returns `true` if the node was NOT previously visited.
    #[inline]
    fn insert(&mut self, id: u32) -> bool {
        match self {
            VisitedSet::Dense { marks, generation } => {
                let idx = id as usize;
                debug_assert!(
                    idx < marks.len(),
                    "VisitedSet::insert: id {} out of bounds (capacity {})",
                    id,
                    marks.len()
                );
                if idx < marks.len() {
                    if marks[idx] != *generation {
                        marks[idx] = *generation;
                        true
                    } else {
                        false
                    }
                } else {
                    // Out-of-bounds: treat as unvisited but don't track.
                    // This shouldn't happen if num_vectors is correct.
                    true
                }
            }
            VisitedSet::Sparse(s) => s.insert(id),
        }
    }

    /// Check if a node has been visited.
    #[inline]
    fn contains(&self, id: u32) -> bool {
        match self {
            VisitedSet::Dense { marks, generation } => {
                let idx = id as usize;
                idx < marks.len() && marks[idx] == *generation
            }
            VisitedSet::Sparse(s) => s.contains(&id),
        }
    }

    /// Prepare for a new search with `num_nodes` total nodes. Reuses the
    /// existing allocation when possible, only reallocating if the index grew.
    fn prepare(&mut self, num_nodes: usize, capacity_hint: usize) {
        match self {
            VisitedSet::Dense { marks, .. } if num_nodes <= DENSE_VISITED_THRESHOLD => {
                if marks.len() < num_nodes {
                    // Index grew: resize and reset
                    marks.resize(num_nodes, 0);
                    // Force a full reset after resize since new slots are 0
                    // and we need generation != 0
                }
                self.clear();
            }
            VisitedSet::Sparse(s) if num_nodes > DENSE_VISITED_THRESHOLD => {
                s.clear();
            }
            _ => {
                // Variant mismatch (index crossed threshold): recreate
                *self = VisitedSet::new(num_nodes, capacity_hint);
            }
        }
    }
}

thread_local! {
    static THREAD_VISITED: RefCell<VisitedSet> = RefCell::new(
        VisitedSet::Dense { marks: Vec::new(), generation: 1 }
    );
}

/// Borrow the thread-local visited set, prepared for `num_nodes`.
/// The closure receives a mutable reference to the reused set.
fn with_visited_set<F, R>(num_nodes: usize, capacity_hint: usize, f: F) -> R
where
    F: FnOnce(&mut VisitedSet) -> R,
{
    THREAD_VISITED.with(|cell| {
        let mut visited = cell.borrow_mut();
        visited.prepare(num_nodes, capacity_hint);
        f(&mut visited)
    })
}

// ─── Candidate types ─────────────────────────────────────────────────────────

/// Candidate node during search.
#[derive(Clone, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) id: u32,
    pub(crate) distance: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller distance = higher priority
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Candidate for min-heap (explore closest first).
#[derive(PartialEq)]
struct MinCandidate {
    id: u32,
    distance: f32,
}
impl Eq for MinCandidate {}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.total_cmp(&self.distance)
    }
}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Result for max-heap (track worst result for pruning).
#[derive(PartialEq)]
struct MaxResult {
    id: u32,
    distance: f32,
}
impl Eq for MaxResult {}
impl Ord for MaxResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.total_cmp(&other.distance)
    }
}
impl PartialOrd for MaxResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─── Search state ────────────────────────────────────────────────────────────

/// Search state for HNSW search algorithm.
pub(crate) struct SearchState {
    /// Candidate queue (min-heap by distance)
    candidates: BinaryHeap<Candidate>,

    /// Visited nodes (to avoid revisiting)
    visited: VisitedSet,

    /// Best distance found so far
    best_distance: f32,

    /// Number of iterations without improvement (for early termination)
    no_improvement_count: usize,
}

impl SearchState {
    /// Create with pre-allocated capacity for better performance.
    pub(crate) fn with_capacity(ef: usize, num_nodes: usize) -> Self {
        Self {
            candidates: BinaryHeap::with_capacity(ef * 2),
            visited: VisitedSet::new(num_nodes, ef * 2),
            best_distance: f32::INFINITY,
            no_improvement_count: 0,
        }
    }

    pub(crate) fn add_candidate(&mut self, id: u32, distance: f32) {
        if !self.visited.contains(id) {
            self.candidates.push(Candidate { id, distance });
        }
    }

    pub(crate) fn pop_candidate(&mut self) -> Option<Candidate> {
        while let Some(candidate) = self.candidates.pop() {
            // Single insert call: returns true if newly visited
            if self.visited.insert(candidate.id) {
                if candidate.distance < self.best_distance {
                    self.best_distance = candidate.distance;
                    self.no_improvement_count = 0;
                } else {
                    self.no_improvement_count += 1;
                }
                return Some(candidate);
            }
        }
        None
    }
}

// ─── Search functions ────────────────────────────────────────────────────────

/// Greedy search in a single layer using standard HNSW beam search.
///
/// Implements the correct HNSW search from Malkov & Yashunin (2016):
/// - Uses min-heap for candidates (explore closest first)
/// - Uses max-heap for results (track worst result for pruning)
/// - Continues until best unexplored candidate is worse than worst result
///
/// This is critical for achieving high recall (~98% on standard benchmarks).
#[cfg(feature = "hnsw")]
pub fn greedy_search_layer(
    query: &[f32],
    entry_point: u32,
    layer: &crate::hnsw::graph::Layer,
    vectors: &[f32],
    dimension: usize,
    ef: usize,
) -> Vec<(u32, f32)> {
    let num_vectors = vectors.len() / dimension;

    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        // Start from entry point
        let entry_vector = get_vector(vectors, dimension, entry_point as usize);
        let entry_distance = cosine_distance(query, entry_vector);
        candidates.push(MinCandidate {
            id: entry_point,
            distance: entry_distance,
        });
        results.push(MaxResult {
            id: entry_point,
            distance: entry_distance,
        });
        visited.insert(entry_point);

        // Standard HNSW beam search:
        // Continue while we have candidates that might improve results
        while let Some(candidate) = candidates.pop() {
            // Stopping condition: if best candidate is worse than worst result
            // and we have enough results, we're done
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if candidate.distance > worst_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            let neighbors = layer.get_neighbors(candidate.id);
            for (i, &neighbor_id) in neighbors.iter().enumerate() {
                // Prefetch next neighbor's vector while processing current
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        prefetch_read_data(vectors.as_ptr().wrapping_add(next_id * dimension));
                    }
                }
                if visited.insert(neighbor_id) {
                    let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
                    let neighbor_distance = cosine_distance(query, neighbor_vector);

                    // Only add if potentially useful
                    let worst_dist =
                        results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                    if results.len() < ef || neighbor_distance < worst_dist {
                        candidates.push(MinCandidate {
                            id: neighbor_id,
                            distance: neighbor_distance,
                        });
                        results.push(MaxResult {
                            id: neighbor_id,
                            distance: neighbor_distance,
                        });

                        // Prune results if over capacity
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to sorted output
        let mut output: Vec<(u32, f32)> =
            results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_by(|a, b| a.1.total_cmp(&b.1));
        output
    })
}

/// Greedy search with adaptive early termination.
///
/// Same beam search as [`greedy_search_layer`], but uses an
/// `EarlyTerminationOracle` to stop once the distance
/// distribution suggests further exploration is unlikely to improve the top-k.
///
/// Returns `(results, num_evaluated)` so callers can inspect how many
/// distance computations were performed.
#[cfg(feature = "hnsw")]
pub fn greedy_search_layer_adaptive(
    query: &[f32],
    entry_point: u32,
    layer: &crate::hnsw::graph::Layer,
    vectors: &[f32],
    dimension: usize,
    ef: usize,
    k: usize,
    config: &crate::adaptive::AdaptiveConfig,
) -> (Vec<(u32, f32)>, usize) {
    use crate::adaptive::EarlyTerminationOracle;

    let num_vectors = vectors.len() / dimension;

    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);
        let mut oracle = EarlyTerminationOracle::new(k, config.clone());

        // Start from entry point
        let entry_vector = get_vector(vectors, dimension, entry_point as usize);
        let entry_distance = cosine_distance(query, entry_vector);
        oracle.observe(entry_distance);

        candidates.push(MinCandidate {
            id: entry_point,
            distance: entry_distance,
        });
        results.push(MaxResult {
            id: entry_point,
            distance: entry_distance,
        });
        visited.insert(entry_point);

        while let Some(candidate) = candidates.pop() {
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if candidate.distance > worst_dist && results.len() >= ef {
                break;
            }

            // Explore neighbors
            let neighbors = layer.get_neighbors(candidate.id);
            for (i, &neighbor_id) in neighbors.iter().enumerate() {
                // Prefetch next neighbor's vector while processing current
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        prefetch_read_data(vectors.as_ptr().wrapping_add(next_id * dimension));
                    }
                }
                if visited.insert(neighbor_id) {
                    let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
                    let neighbor_distance = cosine_distance(query, neighbor_vector);
                    oracle.observe(neighbor_distance);

                    let worst_dist =
                        results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                    if results.len() < ef || neighbor_distance < worst_dist {
                        candidates.push(MinCandidate {
                            id: neighbor_id,
                            distance: neighbor_distance,
                        });
                        results.push(MaxResult {
                            id: neighbor_id,
                            distance: neighbor_distance,
                        });

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }

            // After exploring this candidate's neighbors, check early termination
            if oracle.should_terminate() && results.len() >= k {
                break;
            }
        }

        let num_evaluated = oracle.num_evaluated();
        let mut output: Vec<(u32, f32)> =
            results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_by(|a, b| a.1.total_cmp(&b.1));
        (output, num_evaluated)
    })
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate {
            id: 0,
            distance: 0.5,
        });
        heap.push(Candidate {
            id: 1,
            distance: 0.1,
        });
        heap.push(Candidate {
            id: 2,
            distance: 0.3,
        });

        // Should pop in order: 0.1, 0.3, 0.5 (min-heap)
        assert_eq!(heap.pop().unwrap().distance, 0.1);
        assert_eq!(heap.pop().unwrap().distance, 0.3);
        assert_eq!(heap.pop().unwrap().distance, 0.5);
    }

    #[test]
    fn test_visited_set_dense() {
        let mut v = VisitedSet::new(100, 10);
        assert!(!v.contains(5));
        assert!(v.insert(5));
        assert!(v.contains(5));
        assert!(!v.insert(5)); // already visited
    }

    #[test]
    fn test_visited_set_dense_clear() {
        let mut v = VisitedSet::new(100, 10);
        assert!(v.insert(5));
        assert!(v.contains(5));
        v.clear();
        // After clear, previously visited nodes are no longer marked
        assert!(!v.contains(5));
        assert!(v.insert(5));
    }

    #[test]
    fn test_visited_set_dense_generation_overflow() {
        // Create a dense set and force generation to u16::MAX
        let mut v = VisitedSet::new(100, 10);
        if let VisitedSet::Dense {
            ref mut generation, ..
        } = v
        {
            *generation = u16::MAX;
        }
        assert!(v.insert(5));
        assert!(v.contains(5));
        // This clear triggers the overflow path (memset)
        v.clear();
        assert!(!v.contains(5));
        assert!(v.insert(5));
        assert!(v.contains(5));
    }

    #[test]
    fn test_visited_set_sparse() {
        // Force sparse by using a large num_nodes
        let mut v = VisitedSet::new(DENSE_VISITED_THRESHOLD + 1, 10);
        assert!(!v.contains(42));
        assert!(v.insert(42));
        assert!(v.contains(42));
        assert!(!v.insert(42));
    }

    #[test]
    fn test_visited_set_sparse_clear() {
        let mut v = VisitedSet::new(DENSE_VISITED_THRESHOLD + 1, 10);
        assert!(v.insert(42));
        v.clear();
        assert!(!v.contains(42));
        assert!(v.insert(42));
    }
}
