//! Evaluation metrics for ANN quality.
//!
//! Standard metrics for measuring retrieval quality:
//! - Recall@k: fraction of true neighbors found
//! - Precision@k: fraction of retrieved items that are true neighbors
//! - Mean Average Precision (MAP)
//! - Normalized Discounted Cumulative Gain (NDCG)

use std::collections::HashSet;

/// Compute recall@k: fraction of true k-nearest neighbors that were retrieved.
///
/// recall@k = |retrieved ∩ ground_truth| / k
///
/// # Arguments
///
/// * `ground_truth` - True k-nearest neighbor IDs
/// * `retrieved` - Retrieved neighbor IDs (may be more or fewer than k)
/// * `k` - Number of neighbors we're evaluating
///
/// # Returns
///
/// Recall value in [0.0, 1.0]
pub fn recall_at_k(ground_truth: &[u32], retrieved: &[u32], k: usize) -> f32 {
    if k == 0 || ground_truth.is_empty() {
        return 0.0;
    }

    let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let retrieved_set: HashSet<u32> = retrieved.iter().take(k).copied().collect();

    let intersection = gt_set.intersection(&retrieved_set).count();
    intersection as f32 / k as f32
}

/// Compute precision@k: fraction of retrieved items that are true neighbors.
///
/// precision@k = |retrieved ∩ ground_truth| / |retrieved|
///
/// # Arguments
///
/// * `ground_truth` - True k-nearest neighbor IDs
/// * `retrieved` - Retrieved neighbor IDs
/// * `k` - Number of neighbors to consider
///
/// # Returns
///
/// Precision value in [0.0, 1.0]
pub fn precision_at_k(ground_truth: &[u32], retrieved: &[u32], k: usize) -> f32 {
    if retrieved.is_empty() {
        return 0.0;
    }

    let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let retrieved_k: Vec<u32> = retrieved.iter().take(k).copied().collect();

    let hits = retrieved_k.iter().filter(|id| gt_set.contains(id)).count();
    hits as f32 / retrieved_k.len() as f32
}

/// Compute mean recall across multiple queries.
pub fn mean_recall(ground_truths: &[Vec<u32>], retrievals: &[Vec<u32>], k: usize) -> f32 {
    if ground_truths.is_empty() {
        return 0.0;
    }

    let total: f32 = ground_truths
        .iter()
        .zip(retrievals.iter())
        .map(|(gt, ret)| recall_at_k(gt, ret, k))
        .sum();

    total / ground_truths.len() as f32
}

/// Compute recall at multiple k values.
///
/// Returns recall@1, recall@10, recall@100, etc.
pub fn recall_curve(
    ground_truth: &[u32],
    retrieved: &[u32],
    k_values: &[usize],
) -> Vec<(usize, f32)> {
    k_values
        .iter()
        .map(|&k| (k, recall_at_k(ground_truth, retrieved, k)))
        .collect()
}

/// Evaluation result for a single query.
#[derive(Debug, Clone)]
pub struct QueryEvaluation {
    /// Recall score for this query.
    pub recall: f32,
    /// Precision score for this query.
    pub precision: f32,
    /// Query latency in microseconds.
    pub latency_us: u64,
    /// Number of distance computations (if tracked).
    pub n_distance_computations: Option<usize>,
}

/// Aggregated evaluation results.
#[derive(Debug, Clone)]
pub struct EvaluationSummary {
    /// Number of queries evaluated.
    pub n_queries: usize,
    /// Number of nearest neighbors per query.
    pub k: usize,

    /// Mean recall across all queries.
    pub mean_recall: f32,
    /// Minimum recall observed.
    pub min_recall: f32,
    /// Maximum recall observed.
    pub max_recall: f32,
    /// Standard deviation of recall.
    pub recall_std: f32,

    /// Mean query latency in microseconds.
    pub mean_latency_us: f64,
    /// Median (p50) query latency in microseconds.
    pub p50_latency_us: u64,
    /// 95th-percentile query latency in microseconds.
    pub p95_latency_us: u64,
    /// 99th-percentile query latency in microseconds.
    pub p99_latency_us: u64,
    /// Maximum query latency in microseconds.
    pub max_latency_us: u64,

    /// Queries per second (throughput).
    pub qps: f64,
}

impl EvaluationSummary {
    /// Compute summary statistics from individual query evaluations.
    pub fn from_evaluations(evaluations: &[QueryEvaluation], k: usize) -> Self {
        let n = evaluations.len();
        if n == 0 {
            return Self {
                n_queries: 0,
                k,
                mean_recall: 0.0,
                min_recall: 0.0,
                max_recall: 0.0,
                recall_std: 0.0,
                mean_latency_us: 0.0,
                p50_latency_us: 0,
                p95_latency_us: 0,
                p99_latency_us: 0,
                max_latency_us: 0,
                qps: 0.0,
            };
        }

        // Recall stats
        let recalls: Vec<f32> = evaluations.iter().map(|e| e.recall).collect();
        let mean_recall = recalls.iter().sum::<f32>() / n as f32;
        let min_recall = recalls.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_recall = recalls.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let recall_variance: f32 = recalls
            .iter()
            .map(|r| (r - mean_recall).powi(2))
            .sum::<f32>()
            / n as f32;
        let recall_std = recall_variance.sqrt();

        // Latency stats
        let mut latencies: Vec<u64> = evaluations.iter().map(|e| e.latency_us).collect();
        latencies.sort_unstable();

        let mean_latency_us = latencies.iter().sum::<u64>() as f64 / n as f64;
        let p50_latency_us = latencies[n / 2];
        let p95_latency_us = latencies[(n as f64 * 0.95) as usize];
        let p99_latency_us = latencies[(n as f64 * 0.99) as usize];
        let max_latency_us = latencies.last().copied().unwrap_or(0);

        // QPS (queries per second)
        let total_time_us: u64 = latencies.iter().sum();
        let qps = if total_time_us > 0 {
            n as f64 / (total_time_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        Self {
            n_queries: n,
            k,
            mean_recall,
            min_recall,
            max_recall,
            recall_std,
            mean_latency_us,
            p50_latency_us,
            p95_latency_us,
            p99_latency_us,
            max_latency_us,
            qps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let gt = vec![1, 2, 3, 4, 5];
        let retrieved = vec![1, 2, 3, 6, 7];
        assert!((recall_at_k(&gt, &retrieved, 5) - 0.6).abs() < 0.001);

        // Perfect recall
        let perfect = vec![1, 2, 3, 4, 5];
        assert!((recall_at_k(&gt, &perfect, 5) - 1.0).abs() < 0.001);

        // Zero recall
        let miss = vec![6, 7, 8, 9, 10];
        assert!((recall_at_k(&gt, &miss, 5) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_precision_at_k() {
        let gt = vec![1, 2, 3, 4, 5];
        let retrieved = vec![1, 2, 6, 7, 8];
        assert!((precision_at_k(&gt, &retrieved, 5) - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_recall_curve() {
        let gt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let retrieved = vec![1, 2, 3, 11, 12, 6, 7, 13, 14, 15];

        let curve = recall_curve(&gt, &retrieved, &[1, 5, 10]);
        assert_eq!(curve.len(), 3);
        assert!((curve[0].1 - 1.0).abs() < 0.001); // recall@1 = 1.0
        assert!((curve[1].1 - 0.6).abs() < 0.001); // recall@5 = 3/5
    }

    #[test]
    fn test_evaluation_summary() {
        let evals = vec![
            QueryEvaluation {
                recall: 0.9,
                precision: 0.9,
                latency_us: 100,
                n_distance_computations: None,
            },
            QueryEvaluation {
                recall: 0.8,
                precision: 0.8,
                latency_us: 200,
                n_distance_computations: None,
            },
            QueryEvaluation {
                recall: 1.0,
                precision: 1.0,
                latency_us: 50,
                n_distance_computations: None,
            },
        ];

        let summary = EvaluationSummary::from_evaluations(&evals, 10);
        assert_eq!(summary.n_queries, 3);
        assert!((summary.mean_recall - 0.9).abs() < 0.001);
        assert!((summary.min_recall - 0.8).abs() < 0.001);
        assert!((summary.max_recall - 1.0).abs() < 0.001);
    }
}
