//! Flat NSW graph construction.

use crate::simd;
use crate::RetrieveError;
use smallvec::SmallVec;

use super::graph::NSWIndex;

/// Construct flat NSW graph.
///
/// Uses RNG-based neighbor selection similar to HNSW but without hierarchy.
pub fn construct_graph(index: &mut NSWIndex) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }

    // Initialize neighbor lists
    index.neighbors = vec![SmallVec::new(); index.num_vectors];

    // Set entry point (first vector) if not already set
    if index.entry_point.is_none() {
        index.entry_point = Some(0);
    }

    // Build graph by inserting each vector
    for current_id in 0..index.num_vectors {
        let current_vector = index.get_vector(current_id);

        // Find candidates: all other vectors
        let mut candidates = Vec::new();
        for other_id in 0..index.num_vectors {
            if other_id == current_id {
                continue;
            }

            let other_vector = index.get_vector(other_id);
            let dist = 1.0 - simd::dot(current_vector, other_vector);
            candidates.push((other_id as u32, dist));
        }

        // Select neighbors using RNG-based selection (similar to HNSW)
        let selected = select_neighbors_rng(&candidates, index.params.m);

        // Add bidirectional connections
        for &neighbor_id in &selected {
            // Add connection from current to neighbor
            index.neighbors[current_id].push(neighbor_id);

            // Add reverse connection (if not already present)
            let reverse_neighbors = &mut index.neighbors[neighbor_id as usize];
            if !reverse_neighbors.contains(&(current_id as u32)) {
                reverse_neighbors.push(current_id as u32);
            }
        }
    }

    Ok(())
}

/// Select neighbors using RNG-based selection.
///
/// Similar to HNSW's RNG selection but for flat graph.
fn select_neighbors_rng(candidates: &[(u32, f32)], m: usize) -> Vec<u32> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort by distance
    let mut sorted = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    // RNG-based selection: prefer closer neighbors but allow some randomness
    let mut selected = Vec::new();
    let mut rng = rand::rng();

    // Always include closest
    if !sorted.is_empty() {
        selected.push(sorted[0].0);
    }

    // Select remaining with distance-biased probability
    use rand::Rng;
    for (id, dist) in sorted.iter().skip(1) {
        if selected.len() >= m {
            break;
        }

        // Probability decreases with distance
        let prob = (-dist).exp().min(1.0);
        if rng.random::<f32>() < prob {
            selected.push(*id);
        }
    }

    // If we still need more, add closest remaining
    while selected.len() < m && selected.len() < sorted.len() {
        for (id, _) in &sorted {
            if !selected.contains(id) {
                selected.push(*id);
                break;
            }
        }
    }

    selected
}
