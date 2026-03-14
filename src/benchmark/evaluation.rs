//! Evaluation metrics and infrastructure for ANN benchmarks.
//!
//! Standard metrics used in ANN literature:
//!
//! | Metric | Formula | Interpretation |
//! |--------|---------|----------------|
//! | Recall@K | \|approx ∩ true\| / K | Fraction of true neighbors found |
//! | Precision@K | \|approx ∩ true\| / \|approx\| | Same as recall when \|approx\| = K |
//! | MRR | 1 / rank_of_first_true | Reciprocal of first relevant result |
//! | QPS | queries / seconds | Throughput |
//!
//! # Standard Benchmark Datasets
//!
//! | Dataset | Size | Dim | Distance | Source |
//! |---------|------|-----|----------|--------|
//! | SIFT-1M | 1M | 128 | L2 | INRIA Texmex |
//! | GIST-1M | 1M | 960 | L2 | INRIA Texmex |
//! | GloVe-100 | 1.2M | 100 | Angular | Stanford NLP |
//! | Fashion-MNIST | 60K | 784 | L2 | Zalando |
//! | Deep1B | 1B | 96 | Angular | Yandex |
//!
//! Reference: <https://ann-benchmarks.com/>

use std::collections::HashSet;
use std::time::{Duration, Instant};

pub use crate::distance::{
    angular_distance, cosine_distance, inner_product_distance, l2_distance, normalize,
    DistanceMetric,
};

/// A dataset with ground truth for evaluation.
#[derive(Debug, Clone)]
pub struct EvalDataset {
    /// Name of the dataset
    pub name: String,
    /// Database vectors to index
    pub base: Vec<Vec<f32>>,
    /// Query vectors
    pub queries: Vec<Vec<f32>>,
    /// Ground truth: for each query, the indices of k nearest neighbors
    pub ground_truth: Vec<Vec<u32>>,
    /// Distance metric used for ground truth
    pub metric: DistanceMetric,
    /// Number of neighbors in ground truth
    pub k: usize,
    /// Dimensionality
    pub dim: usize,
}

impl EvalDataset {
    /// Number of base vectors.
    pub fn n_base(&self) -> usize {
        self.base.len()
    }

    /// Number of queries.
    pub fn n_queries(&self) -> usize {
        self.queries.len()
    }

    /// Validate dataset consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.base.is_empty() {
            return Err("Base vectors empty".into());
        }
        if self.queries.is_empty() {
            return Err("Queries empty".into());
        }
        if self.ground_truth.len() != self.queries.len() {
            return Err(format!(
                "Ground truth count {} != query count {}",
                self.ground_truth.len(),
                self.queries.len()
            ));
        }
        for (i, gt) in self.ground_truth.iter().enumerate() {
            if gt.len() < self.k {
                return Err(format!(
                    "Query {} has {} neighbors, expected {}",
                    i,
                    gt.len(),
                    self.k
                ));
            }
        }
        Ok(())
    }
}

/// Evaluation results for a single run.
#[derive(Debug, Clone)]
pub struct EvalResults {
    /// Dataset name
    pub dataset: String,
    /// Algorithm name
    pub algorithm: String,
    /// Configuration string (e.g., "M=16,ef=50")
    pub config: String,
    /// Recall@K for each query
    pub recalls: Vec<f32>,
    /// Query latencies
    pub latencies_us: Vec<u64>,
    /// Index build time
    pub build_time: Duration,
    /// Index memory (approximate)
    pub index_memory_bytes: usize,
    /// K value used
    pub k: usize,
}

impl EvalResults {
    /// Mean recall across all queries.
    pub fn mean_recall(&self) -> f32 {
        if self.recalls.is_empty() {
            return 0.0;
        }
        self.recalls.iter().sum::<f32>() / self.recalls.len() as f32
    }

    /// Median recall.
    pub fn median_recall(&self) -> f32 {
        if self.recalls.is_empty() {
            return 0.0;
        }
        let mut sorted = self.recalls.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Queries per second.
    pub fn qps(&self) -> f64 {
        if self.latencies_us.is_empty() {
            return 0.0;
        }
        let total_us: u64 = self.latencies_us.iter().sum();
        if total_us == 0 {
            return f64::INFINITY;
        }
        self.latencies_us.len() as f64 / (total_us as f64 / 1_000_000.0)
    }

    /// Mean query latency in microseconds.
    pub fn mean_latency_us(&self) -> f64 {
        if self.latencies_us.is_empty() {
            return 0.0;
        }
        self.latencies_us.iter().sum::<u64>() as f64 / self.latencies_us.len() as f64
    }

    /// P50 (median) latency.
    pub fn p50_latency_us(&self) -> u64 {
        if self.latencies_us.is_empty() {
            return 0;
        }
        let mut sorted = self.latencies_us.clone();
        sorted.sort();
        sorted[sorted.len() / 2]
    }

    /// P99 latency.
    pub fn p99_latency_us(&self) -> u64 {
        if self.latencies_us.is_empty() {
            return 0;
        }
        let mut sorted = self.latencies_us.clone();
        sorted.sort();
        sorted[(sorted.len() * 99) / 100]
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        format!(
            "{}[{}]: recall={:.3}, qps={:.1}, p50={:.1}us, build={:.1}s, mem={:.1}MB",
            self.algorithm,
            self.config,
            self.mean_recall(),
            self.qps(),
            self.p50_latency_us(),
            self.build_time.as_secs_f64(),
            self.index_memory_bytes as f64 / 1_000_000.0
        )
    }
}

/// Compute recall@k for a single query.
///
/// Delegates to [`crate::benchmark::metrics::recall_at_k`].
pub fn recall_at_k(approx: &[u32], true_neighbors: &[u32], k: usize) -> f32 {
    crate::benchmark::metrics::recall_at_k(true_neighbors, approx, k)
}

/// Compute mean reciprocal rank for a single query.
///
/// MRR = 1 / (rank of first relevant result)
pub fn mrr(approx: &[u32], true_neighbors: &[u32]) -> f32 {
    let true_set: HashSet<u32> = true_neighbors.iter().copied().collect();
    for (rank, &id) in approx.iter().enumerate() {
        if true_set.contains(&id) {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Evaluate an algorithm on a dataset.
///
/// Generic over any search function.
pub fn evaluate<F>(
    dataset: &EvalDataset,
    algorithm: &str,
    config: &str,
    build_time: Duration,
    index_memory: usize,
    search_fn: F,
) -> EvalResults
where
    F: Fn(&[f32], usize) -> Vec<u32>,
{
    let mut recalls = Vec::with_capacity(dataset.n_queries());
    let mut latencies = Vec::with_capacity(dataset.n_queries());

    for (query, gt) in dataset.queries.iter().zip(dataset.ground_truth.iter()) {
        let start = Instant::now();
        let approx = search_fn(query, dataset.k);
        let elapsed = start.elapsed();

        recalls.push(recall_at_k(&approx, gt, dataset.k));
        latencies.push(elapsed.as_micros() as u64);
    }

    EvalResults {
        dataset: dataset.name.clone(),
        algorithm: algorithm.into(),
        config: config.into(),
        recalls,
        latencies_us: latencies,
        build_time,
        index_memory_bytes: index_memory,
        k: dataset.k,
    }
}

/// Compute ground truth for a dataset.
pub fn compute_ground_truth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(u32, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, vec)| (i as u32, metric.distance(query, vec)))
                .collect();

            distances.sort_by(|a, b| a.1.total_cmp(&b.1));
            distances.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect()
}

// ============ Synthetic Dataset Generators ============

/// Generate a synthetic dataset with uniform random vectors.
pub fn generate_uniform_dataset(
    name: &str,
    n_base: usize,
    n_queries: usize,
    dim: usize,
    k: usize,
    metric: DistanceMetric,
    seed: u64,
) -> EvalDataset {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);

    let base: Vec<Vec<f32>> = (0..n_base)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect();

    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect();

    let ground_truth = compute_ground_truth(&base, &queries, k, metric);

    EvalDataset {
        name: name.into(),
        base,
        queries,
        ground_truth,
        metric,
        k,
        dim,
    }
}

/// Generate a clustered dataset (more realistic).
pub fn generate_clustered_dataset(
    name: &str,
    n_base: usize,
    n_queries: usize,
    dim: usize,
    n_clusters: usize,
    cluster_std: f32,
    k: usize,
    metric: DistanceMetric,
    seed: u64,
) -> EvalDataset {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect();

    // Sample around centers
    let sample_near = |rng: &mut StdRng, center: &[f32]| -> Vec<f32> {
        center
            .iter()
            .map(|&c| {
                let u1: f32 = rng.random();
                let u2: f32 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                (c + z * cluster_std).clamp(0.0, 1.0)
            })
            .collect()
    };

    let base: Vec<Vec<f32>> = (0..n_base)
        .map(|_| {
            let idx = rng.random_range(0..n_clusters);
            sample_near(&mut rng, &centers[idx])
        })
        .collect();

    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| {
            let idx = rng.random_range(0..n_clusters);
            sample_near(&mut rng, &centers[idx])
        })
        .collect();

    let ground_truth = compute_ground_truth(&base, &queries, k, metric);

    EvalDataset {
        name: name.into(),
        base,
        queries,
        ground_truth,
        metric,
        k,
        dim,
    }
}

/// Generate a normalized dataset (for cosine/angular metrics).
pub fn generate_normalized_clustered_dataset(
    name: &str,
    n_base: usize,
    n_queries: usize,
    dim: usize,
    n_clusters: usize,
    cluster_std: f32,
    k: usize,
    seed: u64,
) -> EvalDataset {
    let mut dataset = generate_clustered_dataset(
        name,
        n_base,
        n_queries,
        dim,
        n_clusters,
        cluster_std,
        k,
        DistanceMetric::Cosine,
        seed,
    );

    // Normalize all vectors
    dataset.base = dataset.base.into_iter().map(|v| normalize(&v)).collect();
    dataset.queries = dataset.queries.into_iter().map(|v| normalize(&v)).collect();

    // Recompute ground truth on normalized data
    dataset.ground_truth =
        compute_ground_truth(&dataset.base, &dataset.queries, k, DistanceMetric::Cosine);

    dataset
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let approx = vec![0, 1, 2, 3, 4];
        let truth = vec![0, 2, 4, 6, 8];

        // 3 out of 5 match
        assert!((recall_at_k(&approx, &truth, 5) - 0.6).abs() < 0.01);

        // Perfect recall
        assert!((recall_at_k(&[0, 2, 4, 6, 8], &truth, 5) - 1.0).abs() < 0.01);

        // No overlap
        assert!((recall_at_k(&[1, 3, 5, 7, 9], &truth, 5) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_mrr() {
        let truth = vec![5, 10, 15];

        // First result is relevant
        assert!((mrr(&[5, 1, 2], &truth) - 1.0).abs() < 0.01);

        // Second result is relevant
        assert!((mrr(&[1, 5, 2], &truth) - 0.5).abs() < 0.01);

        // Third result is relevant
        assert!((mrr(&[1, 2, 10], &truth) - 0.333).abs() < 0.01);

        // No relevant results
        assert!((mrr(&[1, 2, 3], &truth) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        // L2 distance: sqrt(2)
        assert!((l2_distance(&a, &b) - 1.414).abs() < 0.01);

        // Cosine distance: orthogonal = 1
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 0.01);

        // Same vector
        assert!((cosine_distance(&a, &a) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 0.01);
        assert!((n[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_eval_dataset_generation() {
        let dataset =
            generate_clustered_dataset("test", 1000, 100, 64, 10, 0.1, 10, DistanceMetric::L2, 42);

        assert_eq!(dataset.n_base(), 1000);
        assert_eq!(dataset.n_queries(), 100);
        assert_eq!(dataset.ground_truth.len(), 100);
        assert_eq!(dataset.ground_truth[0].len(), 10);
        dataset.validate().unwrap();
    }

    #[test]
    fn test_eval_results_summary() {
        let results = EvalResults {
            dataset: "test".into(),
            algorithm: "hnsw".into(),
            config: "M=16".into(),
            recalls: vec![0.8, 0.9, 1.0],
            latencies_us: vec![100, 200, 150],
            build_time: Duration::from_millis(100),
            index_memory_bytes: 1_000_000,
            k: 10,
        };

        assert!((results.mean_recall() - 0.9).abs() < 0.01);
        assert!((results.median_recall() - 0.9).abs() < 0.01);
        assert!(results.qps() > 0.0);
    }
}
