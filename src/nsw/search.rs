//! Flat NSW search algorithm.

use crate::simd;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

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
///
/// Uses a dense generation-counter visited set (O(1) insert/lookup, O(1) clear)
/// instead of HashSet to reduce overhead during construction where this function
/// is called O(n) times.
pub fn greedy_search(
    query: &[f32],
    entry_point: u32,
    neighbors: &[SmallVec<[u32; 16]>],
    vectors: &[f32],
    dimension: usize,
    ef: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    let num_vectors = vectors.len() / dimension;

    // Dense visited array with generation counter.
    // Thread-local reuse across the O(n) calls during construction.
    thread_local! {
        static VISITED: std::cell::RefCell<(Vec<u8>, u8)> =
            const { std::cell::RefCell::new((Vec::new(), 1)) };
    }

    VISITED.with(|cell| {
        let (marks, gen) = &mut *cell.borrow_mut();

        // Resize if index grew
        if marks.len() < num_vectors {
            marks.resize(num_vectors, 0);
        }
        // Advance generation (O(1) clear)
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
            } else if idx >= marks.len() {
                true // out of bounds: treat as unvisited
            } else {
                false
            }
        };

        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

        let entry_vec = get_vector(vectors, dimension, entry_point as usize);
        let entry_dist = 1.0 - simd::dot(query, entry_vec);

        candidates.push(Reverse(Candidate {
            id: entry_point,
            distance: entry_dist,
        }));
        results.push(Candidate {
            id: entry_point,
            distance: entry_dist,
        });
        visited_insert(entry_point);

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            if let Some(neighbor_list) = neighbors.get(current.id as usize) {
                for (i, &neighbor_id) in neighbor_list.iter().enumerate() {
                    // Prefetch next neighbor's vector
                    if i + 1 < neighbor_list.len() {
                        let next_id = neighbor_list[i + 1] as usize;
                        if next_id < num_vectors {
                            let ptr = vectors.as_ptr().wrapping_add(next_id * dimension);
                            // Hint only; no-op if unsupported
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                std::arch::asm!(
                                    "prfm pldl1keep, [{ptr}]",
                                    ptr = in(reg) ptr,
                                    options(nostack, preserves_flags)
                                );
                            }
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                std::arch::x86_64::_mm_prefetch(
                                    ptr as *const i8,
                                    std::arch::x86_64::_MM_HINT_T0,
                                );
                            }
                        }
                    }

                    if !visited_insert(neighbor_id) {
                        continue;
                    }

                    let neighbor_vec = get_vector(vectors, dimension, neighbor_id as usize);
                    let dist = 1.0 - simd::dot(query, neighbor_vec);

                    let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
                    if dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        }));
                        results.push(Candidate {
                            id: neighbor_id,
                            distance: dist,
                        });
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
    })
}

/// Get vector from SoA storage.
#[inline]
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
