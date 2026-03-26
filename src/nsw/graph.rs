//! Flat NSW graph structure.

use crate::RetrieveError;
use smallvec::SmallVec;

/// Flat Navigable Small World index.
///
/// Single-layer graph variant achieving performance parity with HNSW
/// in high-dimensional settings with lower memory overhead.
#[derive(Debug)]
pub struct NSWIndex {
    /// Vectors stored in Structure of Arrays (SoA) format
    pub(crate) vectors: Vec<f32>,

    /// Vector dimension
    pub(crate) dimension: usize,

    /// Number of vectors
    pub(crate) num_vectors: usize,

    /// Single graph layer (no hierarchy)
    pub(crate) neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Parameters
    pub(crate) params: NSWParams,

    /// External doc_ids aligned with internal indices
    doc_ids: Vec<u32>,

    /// Whether index has been built
    built: bool,

    /// Entry point for search
    pub(crate) entry_point: Option<u32>,
}

/// NSW parameters.
#[derive(Clone, Debug)]
pub struct NSWParams {
    /// Maximum number of connections per node (typically 16)
    pub m: usize,

    /// Maximum connections for newly inserted nodes (typically 16)
    pub m_max: usize,

    /// Search width during construction (typically 200)
    pub ef_construction: usize,

    /// Default search width during query (typically 50-200)
    pub ef_search: usize,
}

impl Default for NSWParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl NSWIndex {
    /// Create a new NSW index.
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be greater than 0".to_string(),
            ));
        }
        if m == 0 || m_max == 0 {
            return Err(RetrieveError::InvalidParameter(
                "m and m_max must be greater than 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            neighbors: Vec::new(),
            params: NSWParams {
                m,
                m_max,
                ..Default::default()
            },
            doc_ids: Vec::new(),
            built: false,
            entry_point: None,
        })
    }

    /// Create with custom parameters.
    pub fn with_params(dimension: usize, params: NSWParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be greater than 0".to_string(),
            ));
        }
        if params.m == 0 || params.m_max == 0 {
            return Err(RetrieveError::InvalidParameter(
                "m and m_max must be greater than 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            neighbors: Vec::new(),
            params,
            doc_ids: Vec::new(),
            built: false,
            entry_point: None,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector)
    }

    /// Add a vector to the index from a borrowed slice.
    ///
    /// Notes:
    /// - The index stores vectors internally, so it must copy the slice into its own storage.
    /// - `doc_id` is stored and mapped back in search results.
    pub fn add_slice(&mut self, doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after index is built".into(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        {
            let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
            if (norm_sq - 1.0).abs() > 0.1 {
                return Err(RetrieveError::InvalidParameter(format!(
                    "NSW cosine distance requires L2-normalized vectors \
                     (got norm^2 = {:.4}, expected ~1.0). Use `distance::normalize()`.",
                    norm_sq
                )));
            }
        }

        // Store vector in SoA format
        self.vectors.extend_from_slice(vector);
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;

        Ok(())
    }

    /// Build the index (required before search).
    ///
    /// Constructs the single-layer graph structure.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Construct flat graph
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

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let entry_point = self.entry_point.ok_or(RetrieveError::EmptyIndex)?;

        // Greedy search in single layer
        let results = super::search::greedy_search(
            query,
            entry_point,
            &self.neighbors,
            &self.vectors,
            self.dimension,
            ef.max(k),
        )?;

        // Return top k, mapping internal indices back to external doc_ids
        let mut sorted_results: Vec<(u32, f32)> = results
            .into_iter()
            .take(k)
            .filter_map(|(internal_id, dist)| {
                let doc_id = self.doc_ids.get(internal_id as usize).copied()?;
                Some((doc_id, dist))
            })
            .collect();
        sorted_results.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(sorted_results)
    }

    /// Get vector by index.
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RetrieveError;

    #[test]
    fn test_create_index() {
        let index = NSWIndex::new(4, 8, 8);
        assert!(index.is_ok());
        let index = index.unwrap();
        assert_eq!(index.dimension, 4);
        assert_eq!(index.num_vectors, 0);
    }

    #[test]
    fn test_add_and_search() {
        let mut index = NSWIndex::new(4, 8, 8).unwrap();

        // Add 10 L2-normalized vectors (NSW requires normalized input)
        let raw: Vec<[f32; 4]> = (0..10)
            .map(|i| {
                let v = [i as f32, (i as f32) * 0.5, 1.0, 0.0];
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
                if n > 1e-9 {
                    [v[0] / n, v[1] / n, v[2] / n, v[3] / n]
                } else {
                    [1.0, 0.0, 0.0, 0.0]
                }
            })
            .collect();
        for (i, v) in raw.iter().enumerate() {
            index.add(i as u32, v.to_vec()).unwrap();
        }

        index.build().unwrap();

        // Search for a normalized query close to raw[0] = normalize([0, 0, 1, 0]) = [0, 0, 1, 0]
        let query = vec![0.0, 0.0, 1.0, 0.0];
        let results = index.search(&query, 3, 50).unwrap();

        assert!(!results.is_empty());
        assert!(results.len() <= 3);
        // The closest vector should be doc_id 0 (its normalized form is [0, 0, 1, 0])
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = NSWIndex::new(0, 8, 8);
        assert!(result.is_err());
        match result.unwrap_err() {
            RetrieveError::InvalidParameter(_) => {}
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }
    }
}
