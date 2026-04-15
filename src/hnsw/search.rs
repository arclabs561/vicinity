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

// ─── Visited set ─────────────────────────────────────────────────────────────

/// Threshold below which we use a dense generation-counter array instead of HashSet.
/// 4M nodes = 4MB Vec<u8>, fits in L3 cache on most hardware (L3 is typically 8-32MB).
/// The dense array has O(1) clear (generation bump) vs O(capacity) for HashSet, and
/// avoids hashing overhead on every visited-node check.
const DENSE_VISITED_THRESHOLD: usize = 4_000_000;

/// Fast visited-node tracker using the generation-counter pattern.
///
/// Dense variant: a `Vec<u8>` where `visited[id] == generation` means visited.
/// Incrementing `generation` logically clears the set in O(1). Only when the
/// u8 counter wraps (every 255 searches) does a memset occur.
///
/// Uses u8 (not u16) to minimize cache footprint: 1 byte/node = same as
/// the old Vec<bool>, but with O(1) clear instead of O(n).
///
/// Falls back to `HashSet<u32>` for large indexes where a full dense array
/// would waste memory.
enum VisitedSet {
    Dense { marks: Vec<u8>, generation: u8 },
    Sparse(HashSet<u32>),
}

impl VisitedSet {
    /// Create a visited set sized for `num_nodes` total nodes.
    fn new(num_nodes: usize, capacity_hint: usize) -> Self {
        if num_nodes <= DENSE_VISITED_THRESHOLD {
            VisitedSet::Dense {
                marks: vec![0u8; num_nodes],
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

    /// Check whether a node has been visited.
    #[cfg(test)]
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
    static THREAD_VISITED: RefCell<VisitedSet> = const { RefCell::new(
        VisitedSet::Dense { marks: Vec::new(), generation: 1 }
    ) };
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
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<(u32, f32)> {
    let num_vectors = vectors.len() / dimension;

    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        // Start from entry point
        let entry_vector = get_vector(vectors, dimension, entry_point as usize);
        let entry_distance = dist_fn(query, entry_vector);
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
                // Prefetch upcoming neighbors' vectors to hide DRAM latency.
                // Two cache lines for vectors >64 bytes; 4-ahead for pipeline depth.
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        let ptr = vectors.as_ptr().wrapping_add(next_id * dimension);
                        prefetch_read_data(ptr);
                        prefetch_read_data(ptr.wrapping_add(16));
                    }
                }
                if i + 4 < neighbors.len() {
                    let far_id = neighbors[i + 4] as usize;
                    if far_id < num_vectors {
                        prefetch_read_data(vectors.as_ptr().wrapping_add(far_id * dimension));
                    }
                }
                if visited.insert(neighbor_id) {
                    let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
                    let neighbor_distance = dist_fn(query, neighbor_vector);

                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
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
        }

        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        output
    })
}

/// Greedy search seeded from multiple entry points.
///
/// Same beam search as [`greedy_search_layer`], but initialises the candidate
/// heap with all provided `entry_points`. Duplicate entry IDs are silently
/// deduplicated via the visited set. If `entry_points` is empty this returns
/// an empty result.
///
/// Used by `batch_search_mqo` to warm-start a query from the best result of a
/// nearby query already processed in the batch.
#[cfg(feature = "hnsw")]
pub fn greedy_search_layer_multi_entry(
    query: &[f32],
    entry_points: &[u32],
    layer: &crate::hnsw::graph::Layer,
    vectors: &[f32],
    dimension: usize,
    ef: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> Vec<(u32, f32)> {
    if entry_points.is_empty() {
        return Vec::new();
    }

    let num_vectors = vectors.len() / dimension;

    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        // Seed from every provided entry point.
        for &ep in entry_points {
            if visited.insert(ep) {
                let ep_vec = get_vector(vectors, dimension, ep as usize);
                let ep_dist = dist_fn(query, ep_vec);
                candidates.push(MinCandidate {
                    id: ep,
                    distance: ep_dist,
                });
                results.push(MaxResult {
                    id: ep,
                    distance: ep_dist,
                });
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        while let Some(candidate) = candidates.pop() {
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if candidate.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = layer.get_neighbors(candidate.id);
            for (i, &neighbor_id) in neighbors.iter().enumerate() {
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        let ptr = vectors.as_ptr().wrapping_add(next_id * dimension);
                        prefetch_read_data(ptr);
                        prefetch_read_data(ptr.wrapping_add(16));
                    }
                }
                if i + 4 < neighbors.len() {
                    let far_id = neighbors[i + 4] as usize;
                    if far_id < num_vectors {
                        prefetch_read_data(vectors.as_ptr().wrapping_add(far_id * dimension));
                    }
                }
                if visited.insert(neighbor_id) {
                    let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
                    let neighbor_distance = dist_fn(query, neighbor_vector);

                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
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
        }

        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        output
    })
}

/// Greedy search with a caller-provided distance function.
///
/// Unlike [`greedy_search_layer`] which computes `dist(query, stored_vector)`,
/// this variant takes a closure `dist_fn(query, internal_node_id) -> f32`.
/// The caller is responsible for resolving the node ID to whatever data they
/// need (box geometry, quantized codes, etc.).
///
/// This enables asymmetric distance computation: the graph is built with
/// center-to-center distance for navigability, but search uses any distance
/// function (box-to-point, quantized-to-float, etc.).
///
/// Prefetching is still performed on the stored vector array to keep the
/// center vectors warm for the common case where the custom distance function
/// reads them.
#[cfg(feature = "hnsw")]
pub fn greedy_search_layer_custom<F: Fn(&[f32], u32) -> f32>(
    query: &[f32],
    entry_point: u32,
    layer: &crate::hnsw::graph::Layer,
    vectors: &[f32],
    dimension: usize,
    ef: usize,
    dist_fn: &F,
) -> Vec<(u32, f32)> {
    let num_vectors = vectors.len() / dimension;

    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        // Start from entry point
        let entry_distance = dist_fn(query, entry_point);
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

            let neighbors = layer.get_neighbors(candidate.id);
            for (i, &neighbor_id) in neighbors.iter().enumerate() {
                // Prefetch: keep vectors warm even for custom distance fns
                // that read them (the common case).
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        let ptr = vectors.as_ptr().wrapping_add(next_id * dimension);
                        prefetch_read_data(ptr);
                        prefetch_read_data(ptr.wrapping_add(16));
                    }
                }
                if i + 4 < neighbors.len() {
                    let far_id = neighbors[i + 4] as usize;
                    if far_id < num_vectors {
                        prefetch_read_data(vectors.as_ptr().wrapping_add(far_id * dimension));
                    }
                }
                if visited.insert(neighbor_id) {
                    let neighbor_distance = dist_fn(query, neighbor_id);

                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
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
        }

        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        output
    })
}

/// Greedy search where distance depends on both candidate (parent) and neighbor.
///
/// Used by vertex-relative quantization (SymphonyQGVR) where each edge (u -> v)
/// has a separate quantized code relative to u.
#[cfg(feature = "ivf_rabitq")]
///
/// The closure receives `(parent_id, neighbor_id, neighbor_slot)` where
/// `neighbor_slot` is the index within the parent's neighbor list (for looking
/// up per-edge codes). For the entry point, `parent_id == entry_point` (self-referential).
pub fn greedy_search_layer_edge_aware<F: Fn(u32, u32, usize) -> f32>(
    entry_point: u32,
    entry_dist: f32,
    layer: &crate::hnsw::graph::Layer,
    num_vectors: usize,
    ef: usize,
    dist_fn: &F,
) -> Vec<(u32, f32)> {
    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        candidates.push(MinCandidate {
            id: entry_point,
            distance: entry_dist,
        });
        results.push(MaxResult {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);

        while let Some(candidate) = candidates.pop() {
            let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
            if candidate.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = layer.get_neighbors(candidate.id);
            for (slot, &neighbor_id) in neighbors.iter().enumerate() {
                if visited.insert(neighbor_id) {
                    // The dist_fn closure accesses per-edge data; prefetching
                    // is the caller's responsibility via the closure itself.
                    let neighbor_distance = dist_fn(candidate.id, neighbor_id, slot);

                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
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
        }

        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
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
    dist_fn: fn(&[f32], &[f32]) -> f32,
) -> (Vec<(u32, f32)>, usize) {
    use crate::adaptive::EarlyTerminationOracle;

    let num_vectors = vectors.len() / dimension;

    with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);
        let mut oracle = EarlyTerminationOracle::new(k, config.clone());

        // Start from entry point
        let entry_vector = get_vector(vectors, dimension, entry_point as usize);
        let entry_distance = dist_fn(query, entry_vector);
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
                // Prefetch upcoming neighbors' vectors to hide DRAM latency.
                // Two cache lines for vectors >64 bytes; 4-ahead for pipeline depth.
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        let ptr = vectors.as_ptr().wrapping_add(next_id * dimension);
                        prefetch_read_data(ptr);
                        prefetch_read_data(ptr.wrapping_add(16));
                    }
                }
                if i + 4 < neighbors.len() {
                    let far_id = neighbors[i + 4] as usize;
                    if far_id < num_vectors {
                        prefetch_read_data(vectors.as_ptr().wrapping_add(far_id * dimension));
                    }
                }
                if visited.insert(neighbor_id) {
                    let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
                    let neighbor_distance = dist_fn(query, neighbor_vector);
                    oracle.observe(neighbor_distance);

                    let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
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
        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        (output, num_evaluated)
    })
}

/// Get vector from SoA storage.
#[inline]
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

/// Greedy search with PRT (Probabilistic Routing Test) pre-filtering.
///
/// Before computing the full O(d) distance to a candidate, checks the
/// O(k)-dimensional projected distance. Candidates whose projected distance
/// exceeds the current worst result (scaled by the TFB ratio) are skipped
/// entirely, saving the full distance computation.
///
/// Returns `(results, full_distance_count)`.
#[cfg(feature = "hnsw")]
pub fn greedy_search_layer_prt(
    query: &[f32],
    entry_point: u32,
    layer: &crate::hnsw::graph::Layer,
    vectors: &[f32],
    dimension: usize,
    ef: usize,
    dist_fn: fn(&[f32], &[f32]) -> f32,
    prt: &crate::prt::ProbabilisticRoutingTest,
    query_proj: &[f32],
    tfb: &mut crate::prt::TestFeedbackBuffer,
) -> (Vec<(u32, f32)>, usize) {
    let num_vectors = vectors.len() / dimension;
    let mut full_dist_count: usize = 0;

    let results = with_visited_set(num_vectors, ef * 2, |visited| {
        let mut candidates: BinaryHeap<MinCandidate> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<MaxResult> = BinaryHeap::with_capacity(ef + 1);

        let entry_vector = get_vector(vectors, dimension, entry_point as usize);
        let entry_distance = dist_fn(query, entry_vector);
        full_dist_count += 1;

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

            let neighbors = layer.get_neighbors(candidate.id);
            for (i, &neighbor_id) in neighbors.iter().enumerate() {
                // Prefetch.
                if i + 1 < neighbors.len() {
                    let next_id = neighbors[i + 1] as usize;
                    if next_id < num_vectors {
                        let ptr = vectors.as_ptr().wrapping_add(next_id * dimension);
                        prefetch_read_data(ptr);
                    }
                }
                if !visited.insert(neighbor_id) {
                    continue;
                }

                // PRT pre-filter: skip candidates whose projected distance
                // exceeds the worst result distance (scaled by TFB ratio).
                let worst_dist = results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);
                if results.len() >= ef
                    && !prt.should_compute_full_distance(query_proj, neighbor_id, worst_dist, tfb)
                {
                    continue;
                }

                // Full distance computation.
                let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
                let neighbor_distance = dist_fn(query, neighbor_vector);
                full_dist_count += 1;

                if results.len() < ef || neighbor_distance < worst_dist {
                    tfb.record_true_positive();
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
                } else {
                    tfb.record_false_positive();
                }
            }
        }

        let mut output: Vec<(u32, f32)> = results.into_iter().map(|r| (r.id, r.distance)).collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        output
    });

    (results, full_dist_count)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(MinCandidate {
            id: 0,
            distance: 0.5,
        });
        heap.push(MinCandidate {
            id: 1,
            distance: 0.1,
        });
        heap.push(MinCandidate {
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
        // Create a dense set and force generation to u8::MAX
        let mut v = VisitedSet::new(100, 10);
        if let VisitedSet::Dense {
            ref mut generation, ..
        } = v
        {
            *generation = u8::MAX;
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
