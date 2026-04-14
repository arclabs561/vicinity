//! Vamana search algorithm using beam search.

#[cfg(feature = "vamana")]
use crate::distance as hnsw_distance;
#[cfg(feature = "vamana")]
use crate::vamana::graph::VamanaIndex;
#[cfg(feature = "vamana")]
use crate::RetrieveError;

/// Candidate node during search. Natural ordering: larger distance = greater.
/// Used with `Reverse` for the min-heap (explore closest first).
#[derive(Clone, Copy, PartialEq)]
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

/// Search for k nearest neighbors using beam search.
///
/// Uses a min-heap (closest first) for exploration and collects the
/// nearest `ef` candidates into a result set.
#[cfg(feature = "vamana")]
pub fn search(
    index: &VamanaIndex,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    if query.len() != index.dimension {
        return Err(RetrieveError::DimensionMismatch {
            query_dim: query.len(),
            doc_dim: index.dimension,
        });
    }

    // Dense generation-counter visited set
    thread_local! {
        static VISITED: std::cell::RefCell<(Vec<u8>, u8)> =
            const { std::cell::RefCell::new((Vec::new(), 1)) };
    }

    VISITED.with(|cell| {
        let (marks, gen) = &mut *cell.borrow_mut();
        if marks.len() < index.num_vectors {
            marks.resize(index.num_vectors, 0);
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
            } else if idx >= marks.len() {
                true
            } else {
                false
            }
        };

        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

        let entry_point = index.medoid;
        let entry_vec = index.get_vector(entry_point as usize);
        let entry_dist = hnsw_distance::cosine_distance_normalized(query, entry_vec);

        if entry_dist.is_finite() {
            candidates.push(Reverse(Candidate {
                id: entry_point,
                distance: entry_dist,
            }));
            results.push(Candidate {
                id: entry_point,
                distance: entry_dist,
            });
            visited_insert(entry_point);
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = &index.neighbors[current.id as usize];
            for &neighbor_id in neighbors.iter() {
                if !visited_insert(neighbor_id) {
                    continue;
                }

                let neighbor_vec = index.get_vector(neighbor_id as usize);
                let dist = hnsw_distance::cosine_distance_normalized(query, neighbor_vec);

                if !dist.is_finite() {
                    continue;
                }

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

        let mut output: Vec<(u32, f32)> = results
            .into_iter()
            .map(|c| (index.doc_ids[c.id as usize], c.distance))
            .collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output.into_iter().take(k).collect())
    })
}

/// Search with a custom distance function.
///
/// The closure receives `(query, internal_node_id)` and returns a distance.
/// This enables ADSampling and other asymmetric distance schemes.
#[cfg(feature = "vamana")]
pub fn search_with_distance(
    index: &VamanaIndex,
    query: &[f32],
    k: usize,
    ef: usize,
    dist_fn: &dyn Fn(&[f32], u32) -> f32,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    if query.len() != index.dimension {
        return Err(RetrieveError::DimensionMismatch {
            query_dim: query.len(),
            doc_dim: index.dimension,
        });
    }

    // Dense generation-counter visited set
    thread_local! {
        static VISITED_CUSTOM: std::cell::RefCell<(Vec<u8>, u8)> =
            const { std::cell::RefCell::new((Vec::new(), 1)) };
    }

    VISITED_CUSTOM.with(|cell| {
        let (marks, gen) = &mut *cell.borrow_mut();
        if marks.len() < index.num_vectors {
            marks.resize(index.num_vectors, 0);
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
            } else if idx >= marks.len() {
                true
            } else {
                false
            }
        };

        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef * 2);
        let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

        let entry_point = index.medoid;
        let entry_dist = dist_fn(query, entry_point);

        if entry_dist.is_finite() {
            candidates.push(Reverse(Candidate {
                id: entry_point,
                distance: entry_dist,
            }));
            results.push(Candidate {
                id: entry_point,
                distance: entry_dist,
            });
            visited_insert(entry_point);
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = &index.neighbors[current.id as usize];
            for &neighbor_id in neighbors.iter() {
                if !visited_insert(neighbor_id) {
                    continue;
                }

                let dist = dist_fn(query, neighbor_id);

                if !dist.is_finite() {
                    continue;
                }

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

        let mut output: Vec<(u32, f32)> = results
            .into_iter()
            .map(|c| (index.doc_ids[c.id as usize], c.distance))
            .collect();
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(output.into_iter().take(k).collect())
    })
}
