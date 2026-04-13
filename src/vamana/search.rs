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
    use std::collections::{BinaryHeap, HashSet};

    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    if query.len() != index.dimension {
        return Err(RetrieveError::DimensionMismatch {
            query_dim: query.len(),
            doc_dim: index.dimension,
        });
    }

    let mut visited = HashSet::with_capacity(ef * 2);
    // Min-heap: explore closest candidates first
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef * 2);
    // Max-heap: track worst result for pruning, keep nearest ef
    let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

    // Start from medoid
    let entry_point = index.medoid;
    let entry_vec = index.get_vector(entry_point as usize);
    let entry_dist = hnsw_distance::cosine_distance_normalized(query, entry_vec);

    if entry_dist.is_finite() {
        let entry = Candidate {
            id: entry_point,
            distance: entry_dist,
        };
        candidates.push(Reverse(entry));
        results.push(entry);
        visited.insert(entry_point);
    }

    while let Some(Reverse(current)) = candidates.pop() {
        // Stop when closest unexplored is worse than worst result and we have enough
        let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
        if current.distance > worst_dist && results.len() >= ef {
            break;
        }

        let neighbors = &index.neighbors[current.id as usize];
        for &neighbor_id in neighbors.iter() {
            if !visited.insert(neighbor_id) {
                continue;
            }

            let neighbor_vec = index.get_vector(neighbor_id as usize);
            let dist = hnsw_distance::cosine_distance_normalized(query, neighbor_vec);

            if !dist.is_finite() {
                continue;
            }

            let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
            if dist < worst_dist || results.len() < ef {
                let c = Candidate {
                    id: neighbor_id,
                    distance: dist,
                };
                candidates.push(Reverse(c));
                results.push(c);

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
    use std::collections::{BinaryHeap, HashSet};

    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    if query.len() != index.dimension {
        return Err(RetrieveError::DimensionMismatch {
            query_dim: query.len(),
            doc_dim: index.dimension,
        });
    }

    let mut visited = HashSet::with_capacity(ef * 2);
    let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::with_capacity(ef * 2);
    let mut results: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef + 1);

    let entry_point = index.medoid;
    let entry_dist = dist_fn(query, entry_point);

    if entry_dist.is_finite() {
        let entry = Candidate {
            id: entry_point,
            distance: entry_dist,
        };
        candidates.push(Reverse(entry));
        results.push(entry);
        visited.insert(entry_point);
    }

    while let Some(Reverse(current)) = candidates.pop() {
        let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
        if current.distance > worst_dist && results.len() >= ef {
            break;
        }

        let neighbors = &index.neighbors[current.id as usize];
        for &neighbor_id in neighbors.iter() {
            if !visited.insert(neighbor_id) {
                continue;
            }

            let dist = dist_fn(query, neighbor_id);

            if !dist.is_finite() {
                continue;
            }

            let worst_dist = results.peek().map(|c| c.distance).unwrap_or(f32::INFINITY);
            if dist < worst_dist || results.len() < ef {
                let c = Candidate {
                    id: neighbor_id,
                    distance: dist,
                };
                candidates.push(Reverse(c));
                results.push(c);

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
}
