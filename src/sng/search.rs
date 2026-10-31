//! OPT-SNG search algorithm.

use crate::simd;
use crate::sng::graph::SNGIndex;
use crate::RetrieveError;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Beam width for search; matches the `ef_search` default of the other
/// single-layer graph indexes (NSG, PiPNN).
const SEARCH_EF: usize = 100;

/// Candidate during search.
#[derive(Clone, PartialEq)]
struct SearchCandidate {
    id: u32,
    distance: f32,
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

/// Search OPT-SNG graph for k nearest neighbors.
///
/// Uses greedy search with early termination, leveraging the theoretical
/// guarantee of O(log n) search path length.
pub fn search_sng(
    index: &SNGIndex,
    query: &[f32],
    k: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    if index.num_vectors == 0 {
        return Ok(Vec::new());
    }

    // Start from random entry point (or first vector)
    let mut current = 0u32;
    let mut current_dist = f32::INFINITY;

    // Find good starting point
    for i in 0..index.num_vectors.min(100) {
        let vec = index.get_vector(i);
        let dist = 1.0 - simd::dot(query, vec);
        if dist < current_dist {
            current_dist = dist;
            current = i as u32;
        }
    }

    // Dense generation-counter visited set (O(1) insert/lookup, O(1) clear).
    thread_local! {
        static VISITED: std::cell::RefCell<(Vec<u8>, u8)> =
            const { std::cell::RefCell::new((Vec::new(), 1)) };
    }

    VISITED.with(|cell| {
        let (marks, gen) = &mut *cell.borrow_mut();
        let num_vectors = index.num_vectors;
        if marks.len() < num_vectors {
            marks.resize(num_vectors, 0);
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

        // Best-first beam search. The beam width is independent of k (for
        // k <= SEARCH_EF), so the top-1 of a k=1 query matches the top-1 of
        // a k=10 query. Stopping after k pops, as before, returned the best
        // entry-scan vector with no traversal when k = 1.
        let ef = k.max(SEARCH_EF);
        let mut candidates = BinaryHeap::new();
        // Max-heap on distance: the worst kept result is on top.
        let mut results: BinaryHeap<Reverse<SearchCandidate>> = BinaryHeap::new();

        candidates.push(SearchCandidate {
            id: current,
            distance: current_dist,
        });
        results.push(Reverse(SearchCandidate {
            id: current,
            distance: current_dist,
        }));
        visited_insert(current);

        // Search with O(log n) guarantee
        let max_iterations = (index.num_vectors as f32).ln().ceil() as usize * 10;
        let mut iterations = 0;

        while let Some(candidate) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |r| r.0.distance);
            if results.len() >= ef && candidate.distance > worst {
                break;
            }

            if iterations >= max_iterations {
                break;
            }
            iterations += 1;

            // Explore neighbors
            if let Some(neighbors) = index.neighbors.get(candidate.id as usize) {
                for (i, &neighbor_id) in neighbors.iter().enumerate() {
                    // Prefetch next neighbor's vector
                    if i + 1 < neighbors.len() {
                        let next_id = neighbors[i + 1] as usize;
                        let ptr = index
                            .vectors
                            .as_ptr()
                            .wrapping_add(next_id * index.dimension);
                        crate::prefetch::prefetch_read_data(ptr);
                    }

                    if !visited_insert(neighbor_id) {
                        continue;
                    }

                    let neighbor_vec = index.get_vector(neighbor_id as usize);
                    let dist = 1.0 - simd::dot(query, neighbor_vec);
                    let worst = results.peek().map_or(f32::INFINITY, |r| r.0.distance);
                    if results.len() < ef || dist < worst {
                        let c = SearchCandidate {
                            id: neighbor_id,
                            distance: dist,
                        };
                        candidates.push(c.clone());
                        results.push(Reverse(c));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut results: Vec<(u32, f32)> = results
            .into_iter()
            .map(|Reverse(c)| (c.id, c.distance))
            .collect();

        // Sort by distance and return top k
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(results.into_iter().take(k).collect())
    })
}
