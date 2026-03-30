//! Flat NSW search algorithm.

use crate::simd;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// Candidate for search heaps. Natural ordering: larger distance = greater.
/// Used directly in `results` max-heap (evict farthest), and wrapped in
/// `Reverse` for the `candidates` min-heap (explore closest first).
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
        self.distance.total_cmp(&other.distance)
    }
}

/// Greedy search in flat NSW graph.
///
/// Uses a min-heap (`BinaryHeap<Reverse<Candidate>>`) for the exploration
/// queue (closest first) and a max-heap (`BinaryHeap<Candidate>`) for the
/// result set (evict farthest when full, keeping the nearest ef candidates).
pub fn greedy_search(
    query: &[f32],
    entry_point: u32,
    neighbors: &[SmallVec<[u32; 16]>],
    vectors: &[f32],
    dimension: usize,
    ef: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    let mut visited = HashSet::with_capacity(ef * 2);
    // Min-heap: explore closest candidates first
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef * 2);
    // Max-heap: farthest on top so we can evict it when full
    let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

    // Start from entry point
    let entry_vec = get_vector(vectors, dimension, entry_point as usize);
    let entry_dist = 1.0 - simd::dot(query, entry_vec);

    let entry = Candidate {
        id: entry_point,
        distance: entry_dist,
    };
    candidates.push(Reverse(entry.clone()));
    results.push(entry);
    visited.insert(entry_point);

    while let Some(Reverse(current)) = candidates.pop() {
        // worst_dist = farthest point in results (max-heap peek)
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

                let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
                if dist < worst_dist || results.len() < ef {
                    let c = Candidate {
                        id: neighbor_id,
                        distance: dist,
                    };
                    candidates.push(Reverse(c.clone()));
                    results.push(c);

                    // Evict farthest when over capacity
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
    }

    let mut sorted_results: Vec<(u32, f32)> =
        results.into_iter().map(|c| (c.id, c.distance)).collect();
    sorted_results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    Ok(sorted_results)
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
