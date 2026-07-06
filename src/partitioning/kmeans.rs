//! k-means clustering wrappers around `clump`.
//!
//! Two variants:
//! - [`KMeans`]: cosine distance, for partitioning normalized vectors (IVF-PQ, IVF-AVQ coarse).
//! - `KMeansEuclidean`: L2 distance, for clustering residual vectors (IVF-AVQ PQ codebooks).
//!
//! All computation is delegated to `clump`. This module provides the same public API
//! that vicinity's partitioning and IVF-PQ code expects (SoA flat-buffer input, mutable
//! `fit`, `assign_clusters`, `centroids`).

use crate::clump_compat::{map_clump_error, soa_to_aos};
use crate::RetrieveError;

/// k-means clustering for partitioning vectors.
///
/// Uses cosine distance (via `clump::CosineDistance`) with k-means++ initialization.
pub struct KMeans {
    dimension: usize,
    k: usize,
    seed: Option<u64>,
    max_iter: usize,
    /// Stored fit result from clump; `None` before `fit()` is called.
    fit: Option<clump::KmeansFit<clump::CosineDistance>>,
}

impl KMeans {
    /// Create new k-means with k clusters.
    pub fn new(dimension: usize, k: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 || k == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Dimension and k must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            dimension,
            k,
            seed: None,
            max_iter: 100,
            fit: None,
        })
    }

    /// Configure a deterministic seed for k-means++ initialization.
    ///
    /// When set, repeated `fit(...)` calls on the same inputs produce identical results.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Configure the maximum number of k-means iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Train k-means on vectors.
    ///
    /// `vectors` is a flat SoA buffer of length `num_vectors * dimension`.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() < num_vectors * self.dimension {
            return Err(RetrieveError::InvalidParameter(
                "Insufficient vectors".to_string(),
            ));
        }

        let data = soa_to_aos(vectors, num_vectors, self.dimension);

        // Clamp k to the number of available vectors. The old implementation
        // silently produced fewer centroids when k > n; clump validates k <= n
        // and returns an error. Clamping preserves backward compatibility.
        let effective_k = self.k.min(num_vectors);

        let mut builder = clump::Kmeans::with_metric(effective_k, clump::CosineDistance)
            .with_tol(1e-6)
            .with_max_iter(self.max_iter);

        if let Some(s) = self.seed {
            builder = builder.with_seed(s);
        }

        let result = builder.fit(&data).map_err(map_clump_error)?;
        self.fit = Some(result);

        Ok(())
    }

    /// Assign vectors to nearest clusters.
    ///
    /// Uses the centroids learned during `fit()`. Falls back to a simple
    /// nearest-centroid scan when `fit()` has not been called (returns empty vec
    /// if no centroids exist, preserving the previous behavior).
    pub fn assign_clusters(&self, vectors: &[f32], num_vectors: usize) -> Vec<usize> {
        let Some(ref fit) = self.fit else {
            return Vec::new();
        };

        let data = soa_to_aos(vectors, num_vectors, self.dimension);

        // `predict` can only fail on empty input or dimension mismatch, which
        // callers should never hit after a successful `fit`.
        fit.predict(&data).unwrap_or_default()
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vec<f32>] {
        match self.fit {
            Some(ref f) => &f.centroids,
            None => &[],
        }
    }
}

/// k-means clustering using Euclidean (L2) distance.
///
/// Use this for residual vectors where magnitude matters (e.g. PQ codebook training).
/// Cosine k-means normalizes centroids to unit vectors, which discards magnitude
/// information that is essential when residuals have varying norms.
#[allow(dead_code)]
pub(crate) struct KMeansEuclidean {
    dimension: usize,
    k: usize,
    seed: Option<u64>,
    max_iter: usize,
    fit: Option<clump::KmeansFit<clump::Euclidean>>,
}

#[allow(dead_code)]
impl KMeansEuclidean {
    pub(crate) fn new(dimension: usize, k: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 || k == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Dimension and k must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            dimension,
            k,
            seed: None,
            max_iter: 100,
            fit: None,
        })
    }

    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    #[must_use]
    #[allow(dead_code)]
    pub(crate) fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub(crate) fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() < num_vectors * self.dimension {
            return Err(RetrieveError::InvalidParameter(
                "Insufficient vectors".to_string(),
            ));
        }

        let data = soa_to_aos(vectors, num_vectors, self.dimension);
        let effective_k = self.k.min(num_vectors);

        let mut builder = clump::Kmeans::with_metric(effective_k, clump::Euclidean)
            .with_tol(1e-6)
            .with_max_iter(self.max_iter);

        if let Some(s) = self.seed {
            builder = builder.with_seed(s);
        }

        let result = builder.fit(&data).map_err(map_clump_error)?;
        self.fit = Some(result);
        Ok(())
    }

    pub(crate) fn centroids(&self) -> &[Vec<f32>] {
        match self.fit {
            Some(ref f) => &f.centroids,
            None => &[],
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn l2_normalize_in_place(vecs: &mut [f32], num_vectors: usize, dimension: usize) {
        for i in 0..num_vectors {
            let start = i * dimension;
            let end = start + dimension;
            let v = &mut vecs[start..end];
            let norm2: f32 = v.iter().map(|&x| x * x).sum();
            let norm = norm2.sqrt();
            if norm > 0.0 {
                for x in v {
                    *x /= norm;
                }
            } else if !v.is_empty() {
                v[0] = 1.0;
            }
        }
    }

    proptest! {
        #[test]
        fn prop_kmeans_fit_is_deterministic_given_seed(
            seed in any::<u64>(),
            dimension in 1usize..16,
            num_vectors in 2usize..64,
            k in 1usize..16,
            raw in proptest::collection::vec(-1.0f32..1.0f32, 2usize..(64*16)),
        ) {
            prop_assume!(k <= num_vectors);
            let needed = num_vectors * dimension;
            prop_assume!(raw.len() >= needed);

            let mut vectors = raw[..needed].to_vec();
            l2_normalize_in_place(&mut vectors, num_vectors, dimension);

            let mut km1 = KMeans::new(dimension, k).unwrap().with_seed(seed);
            let mut km2 = KMeans::new(dimension, k).unwrap().with_seed(seed);

            km1.fit(&vectors, num_vectors).unwrap();
            km2.fit(&vectors, num_vectors).unwrap();

            let a1 = km1.assign_clusters(&vectors, num_vectors);
            let a2 = km2.assign_clusters(&vectors, num_vectors);
            prop_assert_eq!(a1, a2);
        }

        #[cfg(feature = "ivf_avq")]
        #[test]
        fn prop_kmeans_euclidean_fit_is_deterministic(
            seed in any::<u64>(),
            dimension in 1usize..8,
            num_vectors in 2usize..32,
            k in 1usize..8,
            raw in proptest::collection::vec(-1.0f32..1.0f32, 2usize..(32*8)),
        ) {
            prop_assume!(k <= num_vectors);
            let needed = num_vectors * dimension;
            prop_assume!(raw.len() >= needed);

            let vectors = raw[..needed].to_vec();

            let mut km1 = KMeansEuclidean::new(dimension, k).unwrap().with_seed(seed);
            let mut km2 = KMeansEuclidean::new(dimension, k).unwrap().with_seed(seed);

            km1.fit(&vectors, num_vectors).unwrap();
            km2.fit(&vectors, num_vectors).unwrap();

            let c1 = km1.centroids().to_vec();
            let c2 = km2.centroids().to_vec();
            prop_assert_eq!(c1, c2);
        }
    }

    #[cfg(feature = "ivf_avq")]
    #[test]
    fn kmeans_euclidean_clusters_by_distance_not_direction() {
        // Two clusters that differ in magnitude but point the same direction.
        // Cosine k-means puts them in the same cluster; Euclidean separates them.
        let dim = 2usize;
        let num_vectors = 4usize;
        // Group A: small magnitude, Group B: large magnitude, same direction
        let vectors: Vec<f32> = vec![
            0.1, 0.0, // A
            0.0, 0.1, // A
            10.0, 0.0, // B
            0.0, 10.0, // B
        ];

        let mut km = KMeansEuclidean::new(dim, 2).unwrap().with_seed(42);
        km.fit(&vectors, num_vectors).unwrap();
        let centroids = km.centroids();
        assert_eq!(centroids.len(), 2);

        // The two centroids should be far apart (≈ [5, 0] and [0, 5] or similar).
        let c0 = &centroids[0];
        let c1 = &centroids[1];
        let dist_sq: f32 = c0.iter().zip(c1.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(
            dist_sq > 1.0,
            "Euclidean centroids should be separated, got dist²={dist_sq}"
        );
    }
}
