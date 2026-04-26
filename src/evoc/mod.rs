//! EVoC (Embedding Vector Oriented Clustering): thin wrapper around `clump::EVoC`.
//!
//! This module re-exports the core clustering types from the `clump` crate and provides
//! a compatibility adapter that accepts vicinity's SoA (flat `&[f32]`) input format.
//!
//! All actual computation is delegated to `clump`.

use crate::clump_compat::{map_clump_error, soa_to_aos};
use crate::RetrieveError;

// Re-export clump types directly.
pub use clump::{ClusterHierarchy, ClusterLayer, ClusterNode};

/// EVoC clustering parameters.
///
/// This is a vicinity-side wrapper that maps to [`clump::EVoCParams`] internally.
/// It preserves the original vicinity parameter names for backward compatibility.
#[derive(Clone, Debug)]
pub struct EVoCParams {
    /// Intermediate dimension for reduction (~15 recommended).
    pub intermediate_dim: usize,

    /// Minimum cluster size (smaller components are treated as noise).
    pub min_cluster_size: usize,

    /// Noise level tolerance (0.0 = cluster more, higher = more noise).
    pub noise_level: f32,

    /// Minimum number of clusters (optional, best-effort).
    pub min_number_clusters: Option<usize>,
}

impl Default for EVoCParams {
    fn default() -> Self {
        Self {
            intermediate_dim: 15,
            min_cluster_size: 10,
            noise_level: 0.0,
            min_number_clusters: None,
        }
    }
}

impl EVoCParams {
    /// Convert to clump's parameter type.
    fn to_clump(&self) -> clump::EVoCParams {
        clump::EVoCParams {
            intermediate_dim: self.intermediate_dim,
            min_cluster_size: self.min_cluster_size,
            noise_level: self.noise_level,
            ..clump::EVoCParams::default()
        }
    }
}

/// EVoC clusterer (thin wrapper around `clump::EVoC`).
///
/// Accepts vicinity's SoA (flat `&[f32]`) input format and delegates to clump
/// for all computation.
pub struct EVoC {
    params: EVoCParams,
    original_dim: usize,
    inner: clump::EVoC,
}

impl EVoC {
    /// Create new EVoC clusterer.
    pub fn new(original_dim: usize, params: EVoCParams) -> Result<Self, RetrieveError> {
        if original_dim == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Original dimension must be greater than 0".to_string(),
            ));
        }

        let inner = clump::EVoC::new(params.to_clump());

        Ok(Self {
            params,
            original_dim,
            inner,
        })
    }

    /// Fit clusterer on vectors and extract hierarchical clusters.
    ///
    /// `vectors` is a flat SoA buffer of length `num_vectors * original_dim`.
    /// Returns the finest-grained layer assignments (noise as `None`).
    pub fn fit_predict(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<Option<usize>>, RetrieveError> {
        let expected_len = num_vectors * self.original_dim;
        if vectors.len() < expected_len {
            return Err(RetrieveError::InvalidParameter(
                "Insufficient vectors".to_string(),
            ));
        }

        // Convert SoA flat buffer to AoS Vec<Vec<f32>>.
        let data = soa_to_aos(vectors, num_vectors, self.original_dim);

        // Delegate to clump.
        let labels = self.inner.fit_predict(&data).map_err(map_clump_error)?;

        // If the caller requested a minimum number of clusters, attempt to refine.
        if let Some(target) = self.params.min_number_clusters {
            if let Ok(layer) = self.inner.layer_for_n_clusters(target) {
                return Ok(layer.assignments);
            }
        }

        Ok(labels)
    }

    /// Get cluster layers (multi-granularity).
    pub fn cluster_layers(&self) -> &[ClusterLayer] {
        self.inner.cluster_layers()
    }

    /// Get cluster hierarchy tree.
    pub fn cluster_tree(&self) -> Option<&ClusterHierarchy> {
        self.inner.cluster_tree()
    }

    /// Get potential duplicate groups.
    pub fn duplicates(&self) -> &[Vec<usize>] {
        self.inner.duplicates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_dimension_error() {
        let result = EVoC::new(0, EVoCParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_fit_predict() {
        let params = EVoCParams {
            intermediate_dim: 2,
            min_cluster_size: 2,
            noise_level: 0.0,
            min_number_clusters: None,
        };
        #[allow(clippy::unwrap_used)]
        let mut evoc = EVoC::new(4, params).unwrap();

        // Create 20 vectors in two distinct clusters
        let mut vectors = Vec::new();
        for i in 0..10 {
            // Cluster A: near [1, 0, 0, 0]
            vectors.extend_from_slice(&[1.0 + i as f32 * 0.01, 0.0, 0.0, 0.0]);
        }
        for i in 0..10 {
            // Cluster B: near [0, 0, 0, 1]
            vectors.extend_from_slice(&[0.0, 0.0, 0.0, 1.0 + i as f32 * 0.01]);
        }

        #[allow(clippy::unwrap_used)]
        let assignments = evoc.fit_predict(&vectors, 20).unwrap();
        assert_eq!(assignments.len(), 20);
    }
}
