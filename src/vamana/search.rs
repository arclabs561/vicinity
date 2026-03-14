//! Vamana search algorithm using beam search.

#[cfg(feature = "vamana")]
use crate::distance as hnsw_distance;
#[cfg(feature = "vamana")]
use crate::vamana::graph::VamanaIndex;
#[cfg(feature = "vamana")]
use crate::RetrieveError;

/// Candidate node during search.
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
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

/// Search for k nearest neighbors using beam search.
///
/// Similar to HNSW but without hierarchy - uses single-layer graph with beam search.
#[cfg(feature = "vamana")]
pub fn search(
    index: &VamanaIndex,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    if query.len() != index.dimension {
        return Err(RetrieveError::DimensionMismatch {
            query_dim: index.dimension,
            doc_dim: query.len(),
        });
    }

    // Use min-heap for candidates (smaller distance = higher priority)
    use std::collections::{BinaryHeap, HashMap};

    // Cache distances to avoid recomputation
    let mut distance_cache: HashMap<u32, f32> = HashMap::with_capacity(ef);
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::with_capacity(ef);

    // Start from random entry point (optimized: use random_range instead of collecting Vec)
    use rand::Rng;
    let mut rng = rand::rng();
    let entry_point = rng.random_range(0..index.num_vectors as u32);

    let entry_vec = index.get_vector(entry_point as usize);
    let entry_dist = hnsw_distance::cosine_distance_normalized(query, entry_vec);

    // Filter out NaN and Infinity
    if entry_dist.is_finite() {
        distance_cache.insert(entry_point, entry_dist);
        candidates.push(Candidate {
            id: entry_point,
            distance: entry_dist,
        });
    }

    // Beam search: explore candidates until we have ef candidates
    // Similar to HNSW: maintain visited set and candidate queue
    let mut visited = std::collections::HashSet::with_capacity(ef);

    while let Some(candidate) = candidates.pop() {
        // Skip if already visited
        if visited.contains(&candidate.id) {
            continue;
        }
        visited.insert(candidate.id);

        // Stop if we have enough candidates (ef limit)
        if visited.len() >= ef {
            break;
        }

        // Explore neighbors
        let neighbors = &index.neighbors[candidate.id as usize];
        for &neighbor_id in neighbors.iter() {
            // Skip if already visited
            if visited.contains(&neighbor_id) {
                continue;
            }

            // Skip if already in distance cache (already computed)
            if distance_cache.contains_key(&neighbor_id) {
                continue;
            }

            let neighbor_vec = index.get_vector(neighbor_id as usize);
            let dist = hnsw_distance::cosine_distance_normalized(query, neighbor_vec);

            // Filter out NaN and Infinity
            if !dist.is_finite() {
                continue;
            }

            // Cache distance and add to candidate queue
            distance_cache.insert(neighbor_id, dist);
            candidates.push(Candidate {
                id: neighbor_id,
                distance: dist,
            });
        }
    }

    // Extract top-k results from cache (already sorted by distance)
    let mut results: Vec<(u32, f32)> = distance_cache.into_iter().collect();
    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    Ok(results.into_iter().take(k).collect())
}
