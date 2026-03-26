//! Synthetic and standard dataset generation for benchmarking.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A dataset for ANN benchmarking.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Training vectors (the database to index)
    pub train: Vec<Vec<f32>>,
    /// Test/query vectors
    pub test: Vec<Vec<f32>>,
    /// Vector dimensionality
    pub dimension: usize,
}

impl Dataset {
    /// Number of training vectors.
    pub fn n_train(&self) -> usize {
        self.train.len()
    }

    /// Number of test vectors.
    pub fn n_test(&self) -> usize {
        self.test.len()
    }

    /// Total memory footprint of raw vectors in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.train.len() + self.test.len()) * self.dimension * std::mem::size_of::<f32>()
    }
}

/// Create a synthetic benchmark dataset with random vectors.
///
/// Vectors are uniformly distributed in [0, 1]^d. This is a baseline
/// dataset - real data often has more structure (clusters, manifolds).
///
/// # Arguments
///
/// * `n_train` - Number of training vectors
/// * `n_test` - Number of test/query vectors
/// * `dimension` - Vector dimensionality
/// * `seed` - Random seed for reproducibility
pub fn create_benchmark_dataset(
    n_train: usize,
    n_test: usize,
    dimension: usize,
    seed: u64,
) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);

    let train: Vec<Vec<f32>> = (0..n_train)
        .map(|_| (0..dimension).map(|_| rng.random::<f32>()).collect())
        .collect();

    let test: Vec<Vec<f32>> = (0..n_test)
        .map(|_| (0..dimension).map(|_| rng.random::<f32>()).collect())
        .collect();

    Dataset {
        train,
        test,
        dimension,
    }
}

/// Create a clustered dataset (more realistic than uniform random).
///
/// Generates `n_clusters` cluster centers, then samples points
/// around each center with Gaussian noise.
///
/// # Arguments
///
/// * `n_train` - Number of training vectors
/// * `n_test` - Number of test/query vectors  
/// * `dimension` - Vector dimensionality
/// * `n_clusters` - Number of clusters
/// * `cluster_std` - Standard deviation within clusters
/// * `seed` - Random seed for reproducibility
pub fn create_clustered_dataset(
    n_train: usize,
    n_test: usize,
    dimension: usize,
    n_clusters: usize,
    cluster_std: f32,
    seed: u64,
) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dimension).map(|_| rng.random::<f32>()).collect())
        .collect();

    // Helper to sample a point near a center
    let sample_near_center = |rng: &mut StdRng, center: &[f32]| -> Vec<f32> {
        center
            .iter()
            .map(|&c| {
                // Box-Muller for Gaussian
                let u1: f32 = rng.random();
                let u2: f32 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                (c + z * cluster_std).clamp(0.0, 1.0)
            })
            .collect()
    };

    let train: Vec<Vec<f32>> = (0..n_train)
        .map(|_| {
            let cluster_idx = rng.random_range(0..n_clusters);
            sample_near_center(&mut rng, &centers[cluster_idx])
        })
        .collect();

    let test: Vec<Vec<f32>> = (0..n_test)
        .map(|_| {
            let cluster_idx = rng.random_range(0..n_clusters);
            sample_near_center(&mut rng, &centers[cluster_idx])
        })
        .collect();

    Dataset {
        train,
        test,
        dimension,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_benchmark_dataset() {
        let dataset = create_benchmark_dataset(100, 10, 64, 42);
        assert_eq!(dataset.n_train(), 100);
        assert_eq!(dataset.n_test(), 10);
        assert_eq!(dataset.dimension, 64);
        assert_eq!(dataset.train[0].len(), 64);
    }

    #[test]
    fn test_create_clustered_dataset() {
        let dataset = create_clustered_dataset(1000, 100, 64, 10, 0.1, 42);
        assert_eq!(dataset.n_train(), 1000);
        assert_eq!(dataset.n_test(), 100);

        // Values should be in [0, 1] range
        for vec in &dataset.train {
            for &v in vec {
                assert!((0.0..=1.0).contains(&v));
            }
        }
    }

    #[test]
    fn test_memory_bytes() {
        let dataset = create_benchmark_dataset(100, 10, 64, 42);
        let expected = (100 + 10) * 64 * 4; // 4 bytes per f32
        assert_eq!(dataset.memory_bytes(), expected);
    }
}
