//! OPT-SNG graph structure.

use crate::RetrieveError;
use smallvec::SmallVec;

/// OPT-SNG index for approximate nearest neighbor search.
///
/// Optimized version of Sparse Neighborhood Graph with:
/// - Automatic truncation parameter optimization
/// - Martingale-based pruning model
/// - Theoretical guarantees
pub struct SNGIndex {
    /// Vectors stored in SoA format
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    /// External doc_ids aligned with internal indices
    doc_ids: Vec<u32>,
    params: SNGParams,
    built: bool,

    /// Graph structure: neighbors[i] = neighbors of vector i
    pub(crate) neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Truncation parameter R (automatically optimized)
    truncation_r: f32,
}

/// OPT-SNG parameters.
#[derive(Clone, Debug)]
pub struct SNGParams {
    /// Maximum out-degree (automatically optimized, but can set initial value)
    pub max_degree: Option<usize>,

    /// Number of hash functions for LSH-based construction (optional)
    pub num_hash_functions: usize,
}

impl Default for SNGParams {
    fn default() -> Self {
        Self {
            max_degree: None, // Will be auto-optimized
            num_hash_functions: 10,
        }
    }
}

impl SNGIndex {
    /// Create a new OPT-SNG index.
    pub fn new(dimension: usize, params: SNGParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            doc_ids: Vec::new(),
            params,
            built: false,
            neighbors: Vec::new(),
            truncation_r: 0.0, // Will be optimized during build
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after index is built".into(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dimension,
                doc_dim: vector.len(),
            });
        }

        self.vectors.extend_from_slice(&vector);
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index with automatic parameter optimization.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Optimize truncation parameter R using closed-form rule
        self.truncation_r =
            crate::sng::optimization::optimize_truncation_r(self.num_vectors, self.dimension)?;

        // Build graph using martingale-based model
        self.construct_graph()?;

        self.built = true;
        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let results = crate::sng::search::search_sng(self, query, k)?;
        // Map internal indices back to external doc_ids
        Ok(results
            .into_iter()
            .filter_map(|(internal_id, dist)| {
                let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                Some((doc_id, dist))
            })
            .collect())
    }

    /// Get vector from SoA storage.
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }

    /// Construct graph using martingale-based model.
    fn construct_graph(&mut self) -> Result<(), RetrieveError> {
        use crate::simd;
        use crate::sng::martingale;
        use smallvec::SmallVec;

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Initialize neighbor lists
        self.neighbors = vec![SmallVec::new(); self.num_vectors];

        // Build graph using martingale-based pruning
        let mut evolution = martingale::CandidateEvolution::new();

        for current_id in 0..self.num_vectors {
            let current_vector = self.get_vector(current_id);

            // Find candidates: all other vectors
            let mut candidates = Vec::new();
            for other_id in 0..self.num_vectors {
                if other_id == current_id {
                    continue;
                }

                let other_vector = self.get_vector(other_id);
                let dist = 1.0 - simd::dot(current_vector, other_vector);
                candidates.push((other_id as u32, dist));
            }

            // Prune using martingale-based model
            let pruned = martingale::prune_candidates_martingale(
                &candidates,
                self.truncation_r,
                &self.vectors,
                self.dimension,
            )?;

            // Update evolution tracker
            evolution.update(pruned.len());

            // Add bidirectional connections
            for &neighbor_id in &pruned {
                // Add connection from current to neighbor
                self.neighbors[current_id].push(neighbor_id);

                // Add reverse connection (if not already present)
                let reverse_neighbors = &mut self.neighbors[neighbor_id as usize];
                if !reverse_neighbors.contains(&(current_id as u32)) {
                    reverse_neighbors.push(current_id as u32);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RetrieveError;

    /// Normalize a vector to unit length (SNG uses dot-product distance).
    fn normalize(v: &mut [f32]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
    }

    #[test]
    fn test_create_index() {
        let index = SNGIndex::new(4, SNGParams::default());
        assert!(index.is_ok());
        let index = index.unwrap();
        assert_eq!(index.dimension, 4);
        assert_eq!(index.num_vectors, 0);
    }

    #[test]
    fn test_add_and_search() {
        let mut index = SNGIndex::new(4, SNGParams::default()).unwrap();

        // Add 10 normalized vectors
        for i in 0..10u32 {
            let mut v = vec![i as f32 + 1.0, (i as f32) * 0.5, 1.0, 0.5];
            normalize(&mut v);
            index.add(i, v).unwrap();
        }

        index.build().unwrap();

        let mut query = vec![1.0, 0.0, 1.0, 0.5];
        normalize(&mut query);
        let results = index.search(&query, 3).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = SNGIndex::new(0, SNGParams::default());
        match result {
            Err(RetrieveError::InvalidParameter(_)) => {}
            Err(other) => panic!("Expected InvalidParameter, got {:?}", other),
            Ok(_) => panic!("Expected error for dimension 0"),
        }
    }
}
