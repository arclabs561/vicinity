//! Parameter auto-tuning for ANN indexes.
//!
//! Inspired by Faiss's AutoTune functionality, this module provides automatic
//! parameter optimization for ANN algorithms. It uses grid search to find
//! optimal parameters based on performance criteria.
//!
//! # Usage
//!
//! ```rust,ignore
//! use vicinity::ann::autotune::{ParameterTuner, Criterion};
//!
//! // Create tuner with recall target
//! let tuner = ParameterTuner::new()
//!     .criterion(Criterion::RecallAtK { k: 10, target: 0.95 })
//!     .time_budget(std::time::Duration::from_secs(60));
//! ```

use crate::benchmark::datasets::{compute_ground_truth, Dataset};
use crate::benchmark::recall_at_k;
#[cfg(feature = "ivf_pq")]
use crate::ivf_pq::IVFPQParams;
use crate::RetrieveError;
use std::time::{Duration, Instant};

/// Performance criterion for auto-tuning.
#[derive(Debug, Clone)]
pub enum Criterion {
    /// Maximize recall@K (target is minimum acceptable recall)
    RecallAtK {
        /// Number of nearest neighbors to evaluate.
        k: usize,
        /// Minimum acceptable recall (e.g., 0.95).
        target: f32,
    },
    /// Minimize query time while maintaining minimum recall
    LatencyWithRecall {
        /// Number of nearest neighbors to evaluate.
        k: usize,
        /// Minimum acceptable recall threshold.
        min_recall: f32,
        /// Maximum acceptable query latency in milliseconds.
        max_latency_ms: f32,
    },
    /// Balance recall and latency (weighted combination)
    Balanced {
        /// Number of nearest neighbors to evaluate.
        k: usize,
        /// Weight for recall in the combined score (0.0 to 1.0).
        recall_weight: f32,
        /// Weight for latency in the combined score (0.0 to 1.0).
        latency_weight: f32,
    },
}

impl Criterion {
    /// Evaluate a parameter configuration.
    ///
    /// Returns a score (higher is better) and whether the criterion is met.
    pub fn evaluate(&self, recall: f32, latency_ms: f32) -> (f32, bool) {
        match self {
            Criterion::RecallAtK { target, .. } => {
                let score = recall;
                let met = recall >= *target;
                (score, met)
            }
            Criterion::LatencyWithRecall {
                min_recall,
                max_latency_ms,
                ..
            } => {
                let recall_met = recall >= *min_recall;
                let latency_met = latency_ms <= *max_latency_ms;
                let met = recall_met && latency_met;
                // Score: negative latency (lower is better), but only if recall is met
                let score = if recall_met {
                    -latency_ms
                } else {
                    recall - 1.0 // Penalize if recall not met
                };
                (score, met)
            }
            Criterion::Balanced {
                recall_weight,
                latency_weight,
                ..
            } => {
                // Normalize: recall [0,1], latency [0, inf] -> normalize to [0,1] range
                // For latency, use inverse (lower is better), cap at reasonable max
                // Use sigmoid-like normalization for better scaling
                let normalized_latency = (latency_ms / 100.0).min(1.0); // Cap at 100ms = 1.0
                let latency_score = 1.0 - normalized_latency;

                // Normalize weights (they should sum to 1.0, but handle if they don't)
                let total_weight = recall_weight + latency_weight;
                let normalized_recall_weight = if total_weight > 0.0 {
                    recall_weight / total_weight
                } else {
                    0.5 // Default to equal weights
                };
                let normalized_latency_weight = if total_weight > 0.0 {
                    latency_weight / total_weight
                } else {
                    0.5
                };

                let score =
                    normalized_recall_weight * recall + normalized_latency_weight * latency_score;
                let met = true; // Balanced always "met" (just optimized)
                (score, met)
            }
        }
    }
}

/// Parameter tuning result.
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Best parameter value found
    pub best_value: usize,
    /// Best score achieved
    pub best_score: f32,
    /// Recall achieved with best parameter
    pub recall: f32,
    /// Latency achieved with best parameter (ms)
    pub latency_ms: f32,
    /// Whether criterion was met
    pub criterion_met: bool,
    /// All parameter values tried
    pub all_results: Vec<(usize, f32, f32, f32)>, // (value, recall, latency, score)
}

/// Parameter tuner for ANN indexes.
pub struct ParameterTuner {
    criterion: Criterion,
    time_budget: Option<Duration>,
    num_test_queries: usize, // Number of queries to use for evaluation
}

impl ParameterTuner {
    /// Create a new parameter tuner.
    pub fn new() -> Self {
        Self {
            criterion: Criterion::RecallAtK {
                k: 10,
                target: 0.95,
            },
            time_budget: None,
            num_test_queries: 100, // Default: use 100 queries for tuning
        }
    }

    /// Set performance criterion.
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Set time budget for tuning (None = no limit).
    pub fn time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = Some(budget);
        self
    }

    /// Set number of test queries to use for evaluation.
    pub fn num_test_queries(mut self, num: usize) -> Self {
        self.num_test_queries = num;
        self
    }

    /// Tune nprobe parameter for IVF-PQ index.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset to use for tuning
    /// * `dimension` - Vector dimension
    /// * `num_clusters` - Number of IVF clusters
    /// * `nprobe_values` - Candidate nprobe values to try
    ///
    /// # Returns
    ///
    /// Optimal nprobe value and performance metrics.
    #[cfg(feature = "ivf_pq")]
    pub fn tune_ivf_pq_nprobe(
        &self,
        dataset: &Dataset,
        dimension: usize,
        num_clusters: usize,
        nprobe_values: &[usize],
    ) -> Result<TuningResult, RetrieveError> {
        // Validate inputs
        if dimension == 0 {
            return Err(RetrieveError::Other(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        if num_clusters == 0 {
            return Err(RetrieveError::Other(
                "num_clusters must be greater than 0".to_string(),
            ));
        }

        if nprobe_values.is_empty() {
            return Err(RetrieveError::Other(
                "nprobe_values cannot be empty".to_string(),
            ));
        }

        if dataset.train.is_empty() {
            return Err(RetrieveError::Other(
                "Dataset training set cannot be empty".to_string(),
            ));
        }

        if dataset.test.is_empty() {
            return Err(RetrieveError::Other(
                "Dataset test set cannot be empty".to_string(),
            ));
        }

        // Validate nprobe values
        for &nprobe in nprobe_values {
            if nprobe == 0 {
                return Err(RetrieveError::Other(
                    "nprobe values must be greater than 0".to_string(),
                ));
            }
            if nprobe > num_clusters {
                return Err(RetrieveError::Other(format!(
                    "nprobe ({}) cannot exceed num_clusters ({})",
                    nprobe, num_clusters
                )));
            }
        }

        let start_time = Instant::now();
        let mut all_results = Vec::new();
        let mut best_value = nprobe_values[0];
        let mut best_score = f32::NEG_INFINITY;
        let mut best_recall = 0.0;
        let mut best_latency = f32::INFINITY;
        let mut criterion_met = false;

        let k = match &self.criterion {
            Criterion::RecallAtK { k, .. } => *k,
            Criterion::LatencyWithRecall { k, .. } => *k,
            Criterion::Balanced { k, .. } => *k,
        };

        // Limit number of test queries
        let num_queries = self.num_test_queries.min(dataset.test.len());
        let test_queries = &dataset.test[..num_queries];

        // Pre-compute ground truth for all queries
        let mut ground_truths = Vec::new();
        for query in test_queries {
            let gt = compute_ground_truth(query, &dataset.train, k);
            ground_truths.push(gt);
        }

        for &nprobe in nprobe_values {
            // Check time budget
            if let Some(budget) = self.time_budget {
                if start_time.elapsed() > budget {
                    break;
                }
            }

            // Create index with this nprobe value
            let params = IVFPQParams {
                num_clusters,
                nprobe,
                num_codebooks: 8,   // Default
                codebook_size: 256, // Default
                use_opq: false,
                #[cfg(feature = "id-compression")]
                id_compression: None,
                #[cfg(feature = "id-compression")]
                compression_threshold: 100,
            };

            let mut index = crate::ivf_pq::IVFPQIndex::new(dimension, params)?;

            // Add vectors
            for (i, vec) in dataset.train.iter().enumerate() {
                index.add(i as u32, vec.clone())?;
            }
            index.build()?;

            // Evaluate on test queries
            let mut recalls = Vec::new();
            let mut latencies = Vec::new();

            for (i, query) in test_queries.iter().enumerate() {
                let query_start = Instant::now();
                let results = index.search(query, k)?;
                let latency = query_start.elapsed().as_secs_f32() * 1000.0; // ms

                let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
                let recall = recall_at_k(&ground_truths[i], &retrieved, k);

                recalls.push(recall);
                latencies.push(latency);
            }

            let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
            let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;

            let (score, met) = self.criterion.evaluate(avg_recall, avg_latency);

            all_results.push((nprobe, avg_recall, avg_latency, score));

            if score > best_score {
                best_score = score;
                best_value = nprobe;
                best_recall = avg_recall;
                best_latency = avg_latency;
                criterion_met = met;
            }
        }

        Ok(TuningResult {
            best_value,
            best_score,
            recall: best_recall,
            latency_ms: best_latency,
            criterion_met,
            all_results,
        })
    }

    /// Tune ef_search parameter for HNSW index.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset to use for tuning
    /// * `dimension` - Vector dimension
    /// * `m` - HNSW m parameter
    /// * `ef_search_values` - Candidate ef_search values to try
    ///
    /// # Returns
    ///
    /// Optimal ef_search value and performance metrics.
    #[cfg(feature = "hnsw")]
    pub fn tune_hnsw_ef_search(
        &self,
        dataset: &Dataset,
        dimension: usize,
        m: usize,
        ef_search_values: &[usize],
    ) -> Result<TuningResult, RetrieveError> {
        // Validate inputs
        if dimension == 0 {
            return Err(RetrieveError::Other(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        if m == 0 {
            return Err(RetrieveError::Other(
                "HNSW m parameter must be greater than 0".to_string(),
            ));
        }

        if ef_search_values.is_empty() {
            return Err(RetrieveError::Other(
                "ef_search_values cannot be empty".to_string(),
            ));
        }

        if dataset.train.is_empty() {
            return Err(RetrieveError::Other(
                "Dataset training set cannot be empty".to_string(),
            ));
        }

        if dataset.test.is_empty() {
            return Err(RetrieveError::Other(
                "Dataset test set cannot be empty".to_string(),
            ));
        }

        // Validate ef_search values
        for &ef_search in ef_search_values {
            if ef_search == 0 {
                return Err(RetrieveError::Other(
                    "ef_search values must be greater than 0".to_string(),
                ));
            }
        }

        let start_time = Instant::now();
        let mut all_results = Vec::new();
        let mut best_value = ef_search_values[0];
        let mut best_score = f32::NEG_INFINITY;
        let mut best_recall = 0.0;
        let mut best_latency = f32::INFINITY;
        let mut criterion_met = false;

        let k = match &self.criterion {
            Criterion::RecallAtK { k, .. } => *k,
            Criterion::LatencyWithRecall { k, .. } => *k,
            Criterion::Balanced { k, .. } => *k,
        };

        // Limit number of test queries
        let num_queries = self.num_test_queries.min(dataset.test.len());
        let test_queries = &dataset.test[..num_queries];

        // Pre-compute ground truth
        let mut ground_truths = Vec::new();
        for query in test_queries {
            let gt = compute_ground_truth(query, &dataset.train, k);
            ground_truths.push(gt);
        }

        // Build index once (ef_search doesn't affect build)
        let mut index = crate::hnsw::HNSWIndex::new(dimension, m, m)?;
        for (i, vec) in dataset.train.iter().enumerate() {
            index.add(i as u32, vec.clone())?;
        }
        index.build()?;

        for &ef_search in ef_search_values {
            // Check time budget
            if let Some(budget) = self.time_budget {
                if start_time.elapsed() > budget {
                    break;
                }
            }

            // Evaluate with this ef_search
            let mut recalls = Vec::new();
            let mut latencies = Vec::new();

            for (i, query) in test_queries.iter().enumerate() {
                let query_start = Instant::now();
                let results = index.search(query, k, ef_search)?;
                let latency = query_start.elapsed().as_secs_f32() * 1000.0; // ms

                let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
                let recall = recall_at_k(&ground_truths[i], &retrieved, k);

                recalls.push(recall);
                latencies.push(latency);
            }

            let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
            let avg_latency = latencies.iter().sum::<f32>() / latencies.len() as f32;

            let (score, met) = self.criterion.evaluate(avg_recall, avg_latency);

            all_results.push((ef_search, avg_recall, avg_latency, score));

            if score > best_score {
                best_score = score;
                best_value = ef_search;
                best_recall = avg_recall;
                best_latency = avg_latency;
                criterion_met = met;
            }
        }

        Ok(TuningResult {
            best_value,
            best_score,
            recall: best_recall,
            latency_ms: best_latency,
            criterion_met,
            all_results,
        })
    }
}

impl Default for ParameterTuner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::benchmark::datasets::create_benchmark_dataset;

    #[test]
    fn test_tuner_creation() {
        let tuner = ParameterTuner::new()
            .criterion(Criterion::RecallAtK {
                k: 10,
                target: 0.95,
            })
            .time_budget(Duration::from_secs(60));
        assert!(matches!(tuner.criterion, Criterion::RecallAtK { .. }));
    }

    #[test]
    fn test_criterion_evaluation() {
        // RecallAtK criterion
        let criterion = Criterion::RecallAtK {
            k: 10,
            target: 0.95,
        };
        let (score, met) = criterion.evaluate(0.97, 10.0);
        assert!(met);
        assert!(score > 0.95);

        let (score2, met2) = criterion.evaluate(0.90, 5.0);
        assert!(!met2);
        let _ = score2; // Verified above

        // LatencyWithRecall criterion
        let criterion = Criterion::LatencyWithRecall {
            k: 10,
            min_recall: 0.90,
            max_latency_ms: 10.0,
        };
        let (score, met) = criterion.evaluate(0.95, 8.0);
        assert!(met);
        assert!(score < 0.0); // Negative latency (lower is better)

        let (_score2, met2) = criterion.evaluate(0.85, 5.0); // Recall too low
        assert!(!met2);

        let (_score3, met3) = criterion.evaluate(0.95, 15.0); // Latency too high
        assert!(!met3);

        // Balanced criterion
        let criterion = Criterion::Balanced {
            k: 10,
            recall_weight: 0.7,
            latency_weight: 0.3,
        };
        let (score, met) = criterion.evaluate(0.95, 10.0);
        assert!(met);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_tune_ivf_pq_nprobe_validation() {
        #[cfg(feature = "ivf_pq")]
        {
            let tuner = ParameterTuner::new();
            let dataset = create_benchmark_dataset(100, 10, 128, 42);

            // Valid case
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 128, 16, &[1, 2, 4, 8]);
            assert!(result.is_ok());

            // Invalid: zero dimension
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 0, 16, &[1, 2, 4, 8]);
            assert!(result.is_err());

            // Invalid: zero clusters
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 128, 0, &[1, 2, 4, 8]);
            assert!(result.is_err());

            // Invalid: empty nprobe values
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 128, 16, &[]);
            assert!(result.is_err());

            // Invalid: nprobe > num_clusters
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 128, 16, &[1, 2, 20]); // 20 > 16
            assert!(result.is_err());

            // Invalid: zero nprobe
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 128, 16, &[0, 1, 2]);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tune_hnsw_ef_search_validation() {
        #[cfg(feature = "hnsw")]
        {
            let tuner = ParameterTuner::new();
            let dataset = create_benchmark_dataset(100, 10, 128, 42);

            // Valid case
            let result = tuner.tune_hnsw_ef_search(&dataset, 128, 16, &[10, 20, 50]);
            assert!(result.is_ok());

            // Invalid: zero dimension
            let result = tuner.tune_hnsw_ef_search(&dataset, 0, 16, &[10, 20, 50]);
            assert!(result.is_err());

            // Invalid: zero m
            let result = tuner.tune_hnsw_ef_search(&dataset, 128, 0, &[10, 20, 50]);
            assert!(result.is_err());

            // Invalid: empty ef_search values
            let result = tuner.tune_hnsw_ef_search(&dataset, 128, 16, &[]);
            assert!(result.is_err());

            // Invalid: zero ef_search
            let result = tuner.tune_hnsw_ef_search(&dataset, 128, 16, &[0, 10, 20]);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tune_empty_dataset() {
        #[cfg(feature = "ivf_pq")]
        {
            let tuner = ParameterTuner::new();
            let empty_dataset = Dataset {
                train: Vec::new(),
                test: Vec::new(),
                dimension: 128,
            };

            let result = tuner.tune_ivf_pq_nprobe(&empty_dataset, 128, 16, &[1, 2, 4]);
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_tune_time_budget() {
        #[cfg(feature = "ivf_pq")]
        {
            let tuner = ParameterTuner::new().time_budget(Duration::from_millis(1)); // Very short budget

            let dataset = create_benchmark_dataset(1000, 100, 128, 42);

            // Should respect time budget and stop early
            let result = tuner.tune_ivf_pq_nprobe(&dataset, 128, 16, &[1, 2, 4, 8, 16, 32, 64]);
            // May succeed or fail depending on timing, but should handle gracefully
            let _ = result; // Just check it doesn't panic
        }
    }

    #[test]
    fn test_tune_with_small_dataset() {
        #[cfg(feature = "ivf_pq")]
        {
            let tuner = ParameterTuner::new().num_test_queries(5); // Very few queries

            let dataset = create_benchmark_dataset(50, 10, 64, 42);

            let result = tuner.tune_ivf_pq_nprobe(&dataset, 64, 8, &[1, 2, 4]);
            assert!(result.is_ok());

            let tuning_result = result.unwrap();
            assert!(!tuning_result.all_results.is_empty());
            assert!(tuning_result.recall >= 0.0 && tuning_result.recall <= 1.0);
            assert!(tuning_result.latency_ms >= 0.0);
        }
    }
}
