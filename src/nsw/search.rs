//! Flat NSW search algorithm.

use crate::simd;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::collections::{BinaryHeap, HashSet};

/// Candidate during search.
#[derive(Clone, PartialEq)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

/// Greedy search in flat NSW graph.
///
/// # Performance Note
///
/// Uses `HashSet` for visited tracking. For very large graphs (>1M nodes) with
/// high ef values, a `BitVec` could reduce memory overhead, but `HashSet` with
/// pre-allocation performs well for typical workloads and doesn't require
/// knowing the total node count upfront.
pub fn greedy_search(
    query: &[f32],
    entry_point: u32,
    neighbors: &[SmallVec<[u32; 16]>],
    vectors: &[f32],
    dimension: usize,
    ef: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    // Pre-allocate with capacity - HashSet is O(1) average for insert/lookup.
    // For dense graphs with known size, BitVec would be more cache-efficient.
    let mut visited = HashSet::with_capacity(ef * 2);
    let mut candidates = BinaryHeap::with_capacity(ef * 2);
    let mut results = BinaryHeap::with_capacity(ef);

    // Start from entry point
    let entry_vec = get_vector(vectors, dimension, entry_point as usize);
    let entry_dist = 1.0 - simd::dot(query, entry_vec);

    candidates.push(Candidate {
        id: entry_point,
        distance: entry_dist,
    });
    results.push(Candidate {
        id: entry_point,
        distance: entry_dist,
    });
    visited.insert(entry_point);

    // Beam search with stopping condition
    while let Some(current) = candidates.pop() {
        // Stop when best unexplored candidate is worse than worst result
        // and we have enough results (standard HNSW/NSW termination)
        let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
        if current.distance > worst_dist && results.len() >= ef {
            break;
        }

        if let Some(neighbor_list) = neighbors.get(current.id as usize) {
            for &neighbor_id in neighbor_list.iter() {
                if visited.contains(&neighbor_id) {
                    continue;
                }

                visited.insert(neighbor_id);

                let neighbor_vec = get_vector(vectors, dimension, neighbor_id as usize);
                let dist = 1.0 - simd::dot(query, neighbor_vec);

                // Add to candidates if better than worst result
                let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
                if dist < worst_dist || results.len() < ef {
                    candidates.push(Candidate {
                        id: neighbor_id,
                        distance: dist,
                    });
                    results.push(Candidate {
                        id: neighbor_id,
                        distance: dist,
                    });

                    // Keep only top ef
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
    }

    // Convert to sorted vector
    let mut sorted_results: Vec<(u32, f32)> =
        results.into_iter().map(|c| (c.id, c.distance)).collect();
    sorted_results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1)); // Unstable for better performance

    Ok(sorted_results)
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
