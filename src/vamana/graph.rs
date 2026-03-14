//! Vamana graph structure and core types.

use crate::RetrieveError;
use smallvec::SmallVec;

#[cfg(feature = "vamana")]
/// Vamana parameters controlling graph structure and search behavior.
#[derive(Clone, Debug)]
pub struct VamanaParams {
    /// Maximum out-degree per node (typically 64-128, higher for SSD serving)
    pub max_degree: usize,

    /// Relaxation factor for RRND (typically 1.3-1.5)
    /// Higher alpha = less pruning, larger graphs
    pub alpha: f32,

    /// Search width during construction (typically 200-400)
    pub ef_construction: usize,

    /// Default search width during query (typically 50-200)
    pub ef_search: usize,
}

#[cfg(feature = "vamana")]
impl Default for VamanaParams {
    fn default() -> Self {
        Self {
            max_degree: 64,
            alpha: 1.3,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

#[cfg(feature = "vamana")]
/// Vamana index for approximate nearest neighbor search.
///
/// Uses two-pass construction with RRND + RND for high-quality graph structure.
pub struct VamanaIndex {
    /// Vector dimension
    pub(crate) dimension: usize,

    /// Vectors stored in Structure of Arrays (SoA) layout
    pub(crate) vectors: Vec<f32>,

    /// Neighbor lists for each vector
    pub(crate) neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Parameters
    pub(crate) params: VamanaParams,

    /// Number of vectors added
    pub(crate) num_vectors: usize,

    /// Whether index has been built
    built: bool,

    /// Medoid (closest point to centroid), used as search entry point.
    /// Computed during build.
    pub(crate) medoid: u32,
}

#[cfg(feature = "vamana")]
impl VamanaIndex {
    /// Create a new Vamana index.
    pub fn new(dimension: usize, params: VamanaParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        Ok(Self {
            dimension,
            vectors: Vec::new(),
            neighbors: Vec::new(),
            params,
            num_vectors: 0,
            built: false,
            medoid: 0,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after build".into(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dimension,
                doc_dim: vector.len(),
            });
        }

        // Extend vectors array (SoA layout)
        self.vectors.extend_from_slice(&vector);
        self.neighbors.push(SmallVec::new());
        self.num_vectors += 1;

        Ok(())
    }

    /// Build the index (two-pass construction).
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "index already built".into(),
            ));
        }

        // Two-pass construction: RRND + RND
        super::construction::construct_graph(self)?;
        self.built = true;

        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }

        super::search::search(self, query, k, ef)
    }

    /// Get vector by index.
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

#[cfg(all(test, feature = "vamana"))]
mod tests {
    use super::*;
    use crate::distance;

    #[test]
    fn test_vamana_create() {
        let params = VamanaParams::default();
        let index = VamanaIndex::new(128, params);
        assert!(index.is_ok());
    }

    #[test]
    fn test_vamana_add() {
        let params = VamanaParams::default();
        let mut index = VamanaIndex::new(128, params).unwrap();

        let vector = vec![0.1; 128];
        assert!(index.add(0, vector).is_ok());
        assert_eq!(index.num_vectors, 1);
    }

    /// Generate `n` random normalized vectors of given dimension using a simple LCG.
    fn generate_normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                let raw: Vec<f32> = (0..dim)
                    .map(|_| {
                        // Simple LCG for reproducibility without extra deps
                        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                        // Map to [-1, 1]
                        ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
                    })
                    .collect();
                distance::normalize(&raw)
            })
            .collect()
    }

    #[test]
    fn test_vamana_build_does_not_panic() {
        let dim = 32;
        let n = 60;
        let vectors = generate_normalized_vectors(n, dim, 42);

        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 40,
            ef_search: 20,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        // Must not panic
        index.build().unwrap();

        // Medoid should be within valid range
        assert!((index.medoid as usize) < n);
    }

    #[test]
    fn test_vamana_search_self_query() {
        let dim = 32;
        let n = 80;
        let vectors = generate_normalized_vectors(n, dim, 99);

        let params = VamanaParams {
            max_degree: 32,
            alpha: 1.5,
            ef_construction: 100,
            ef_search: 80,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // For each of a sample of vectors, the self-query should appear in the
        // top-k results with distance close to zero.
        let k = 5;
        let ef = 80;
        let sample_indices = [0, 1, n / 2, n - 1];
        for &idx in &sample_indices {
            let results = index.search(&vectors[idx], k, ef).unwrap();
            assert!(
                !results.is_empty(),
                "search returned empty results for query {}",
                idx
            );
            let found = results
                .iter()
                .any(|&(id, dist)| id == idx as u32 && dist < 1e-4);
            assert!(
                found,
                "self-query for vector {} not found in top-{} results: {:?}",
                idx, k, results
            );
        }
    }

    #[test]
    fn test_vamana_search_deterministic() {
        let dim = 32;
        let n = 60;
        let vectors = generate_normalized_vectors(n, dim, 77);

        let params = VamanaParams {
            max_degree: 16,
            alpha: 1.3,
            ef_construction: 40,
            ef_search: 30,
        };
        let mut index = VamanaIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();

        // Same query should return same results (medoid entry point is deterministic)
        let query = &vectors[5];
        let r1 = index.search(query, 5, 30).unwrap();
        let r2 = index.search(query, 5, 30).unwrap();
        assert_eq!(
            r1, r2,
            "search should be deterministic with medoid entry point"
        );
    }
}
