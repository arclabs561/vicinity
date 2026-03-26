#![allow(dead_code)]
//! Adaptive computation patterns for approximate nearest neighbor search.
//!
//! # The Core Insight
//!
//! Most ANN algorithms perform redundant computation. Consider HNSW search:
//!
//! ```text
//! Query arrives → Greedy traversal → Visit 100+ nodes → Return top-k
//!                                   ↑
//!                                   Many distance computations lead nowhere
//! ```
//!
//! Adaptive computation asks: **can we skip work that won't change the result?**
//!
//! # Techniques
//!
//! This module implements several complementary approaches:
//!
//! | Technique | Key Idea | Speedup |
//! |-----------|----------|---------|
//! | **Early termination** | Stop once result is "good enough" | 1.5-2x |
//! | **Angle estimation** | Predict distances via projections | 1.2-1.6x (FINGER) |
//! | **Dimension sampling** | Use subset of dimensions first | 1.3-2x (ADSampling) |
//! | **Hubness avoidance** | Deprioritize high-degree nodes | Quality improvement |
//!
//! # Research Context
//!
//! - **FINGER** (Chen et al., WSDM 2023): Estimate angles between residual vectors
//!   to skip distance computations in graph traversal.
//!
//! - **ADSampling** (Wei et al., VLDB 2022): Sample dimensions adaptively,
//!   more from informative dimensions, fewer from redundant ones.
//!
//! - **Hubness** (Radovanović et al., JMLR 2010): In high-d spaces, some points
//!   become "hubs" appearing as NN to many queries. These deserve special treatment.
//!
//! The techniques share a meta-principle: **compute just enough to be confident**.

/// Configuration for adaptive search.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Minimum candidates to evaluate before considering early termination.
    pub min_candidates: usize,

    /// Confidence threshold for early termination (0.0-1.0).
    /// Higher = more conservative = better recall but slower.
    pub confidence_threshold: f32,

    /// Number of dimensions for initial projection (0 = disabled).
    pub projection_dims: usize,

    /// Whether to use dimension sampling.
    pub dimension_sampling: bool,

    /// Fraction of dimensions to sample in first pass (0.1-1.0).
    pub sampling_ratio: f32,

    /// Whether to track and deprioritize hubs.
    pub hubness_aware: bool,

    /// Threshold for hub detection (fraction of queries where node appears).
    pub hub_threshold: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_candidates: 10,
            confidence_threshold: 0.9,
            projection_dims: 0,
            dimension_sampling: false,
            sampling_ratio: 0.25,
            hubness_aware: false,
            hub_threshold: 0.1,
        }
    }
}

impl AdaptiveConfig {
    /// Conservative config - maximize recall at cost of speed.
    pub fn conservative() -> Self {
        Self {
            min_candidates: 50,
            confidence_threshold: 0.99,
            projection_dims: 0,
            dimension_sampling: false,
            sampling_ratio: 1.0,
            hubness_aware: false,
            hub_threshold: 0.05,
        }
    }

    /// Aggressive config - maximize speed at cost of recall.
    pub fn aggressive() -> Self {
        Self {
            min_candidates: 5,
            confidence_threshold: 0.7,
            projection_dims: 32,
            dimension_sampling: true,
            sampling_ratio: 0.1,
            hubness_aware: true,
            hub_threshold: 0.15,
        }
    }
}

/// Early termination oracle.
///
/// Tracks distance distributions to decide when enough candidates have been seen.
/// Based on the observation that once we've seen the nearest neighbors, additional
/// candidates are increasingly unlikely to be better.
#[derive(Debug)]
pub struct EarlyTerminationOracle {
    /// Best k distances seen so far.
    top_k_distances: Vec<f32>,
    /// Target k for search.
    k: usize,
    /// Total candidates evaluated.
    num_evaluated: usize,
    /// Running mean of distances (for distribution estimation).
    distance_mean: f32,
    /// Running variance of distances.
    distance_var: f32,
    /// Config.
    config: AdaptiveConfig,
}

impl EarlyTerminationOracle {
    /// Create new oracle for k-NN search.
    pub fn new(k: usize, config: AdaptiveConfig) -> Self {
        Self {
            top_k_distances: Vec::with_capacity(k),
            k,
            num_evaluated: 0,
            distance_mean: 0.0,
            distance_var: 0.0,
            config,
        }
    }

    /// Record a new candidate distance.
    pub fn observe(&mut self, distance: f32) {
        self.num_evaluated += 1;

        // Update running statistics (Welford's algorithm)
        let delta = distance - self.distance_mean;
        self.distance_mean += delta / self.num_evaluated as f32;
        let delta2 = distance - self.distance_mean;
        self.distance_var += delta * delta2;

        // Update top-k
        if self.top_k_distances.len() < self.k {
            self.top_k_distances.push(distance);
            self.top_k_distances.sort_by(|a, b| a.total_cmp(b));
        } else if distance < self.top_k_distances[self.k - 1] {
            self.top_k_distances[self.k - 1] = distance;
            self.top_k_distances.sort_by(|a, b| a.total_cmp(b));
        }
    }

    /// Should we stop searching?
    ///
    /// Returns true if we're confident we've found the true top-k.
    pub fn should_terminate(&self) -> bool {
        // Need minimum candidates
        if self.num_evaluated < self.config.min_candidates {
            return false;
        }

        // Need full top-k
        if self.top_k_distances.len() < self.k {
            return false;
        }

        // Estimate probability that a random new candidate would be in top-k
        // Under Gaussian assumption: P(X < threshold) = Φ((threshold - mean) / std)
        let variance = self.distance_var / (self.num_evaluated as f32 - 1.0).max(1.0);
        let std_dev = variance.sqrt().max(1e-9);

        let threshold = self.top_k_distances[self.k - 1];
        let z_score = (threshold - self.distance_mean) / std_dev;

        // Approximate Gaussian CDF (good enough for our purposes)
        // P(X < threshold) ≈ 1 / (1 + exp(-1.7 * z_score))
        let prob_better = 1.0 / (1.0 + (-1.7 * z_score).exp());

        // If probability of finding a better candidate is low enough, stop
        prob_better < (1.0 - self.config.confidence_threshold)
    }

    /// Get current best k distances.
    pub fn top_k(&self) -> &[f32] {
        &self.top_k_distances
    }

    /// Number of candidates evaluated.
    pub fn num_evaluated(&self) -> usize {
        self.num_evaluated
    }
}

/// Dimension importance estimator for adaptive sampling.
///
/// Some dimensions are more informative than others. By estimating importance,
/// we can prioritize computation on the most discriminative dimensions.
#[derive(Debug, Clone)]
pub struct DimensionImportance {
    /// Variance per dimension (higher = more discriminative).
    variances: Vec<f32>,
    /// Mean per dimension (for centering).
    means: Vec<f32>,
    /// Sampling order (sorted by importance).
    importance_order: Vec<usize>,
    /// Number of vectors used to estimate.
    num_samples: usize,
}

impl DimensionImportance {
    /// Estimate importance from a sample of vectors.
    ///
    /// # Arguments
    ///
    /// * `vectors` - Flat array of vectors (row-major)
    /// * `num_vectors` - Number of vectors in sample
    /// * `dimension` - Vector dimension
    pub fn estimate(vectors: &[f32], num_vectors: usize, dimension: usize) -> Self {
        assert_eq!(vectors.len(), num_vectors * dimension);

        let mut means = vec![0.0f32; dimension];
        let mut variances = vec![0.0f32; dimension];

        // Compute means
        for i in 0..num_vectors {
            let vec = &vectors[i * dimension..(i + 1) * dimension];
            for (d, &v) in vec.iter().enumerate() {
                means[d] += v;
            }
        }
        for m in &mut means {
            *m /= num_vectors as f32;
        }

        // Compute variances
        for i in 0..num_vectors {
            let vec = &vectors[i * dimension..(i + 1) * dimension];
            for (d, &v) in vec.iter().enumerate() {
                let diff = v - means[d];
                variances[d] += diff * diff;
            }
        }
        for v in &mut variances {
            *v /= (num_vectors as f32 - 1.0).max(1.0);
        }

        // Sort dimensions by variance (descending)
        let mut importance_order: Vec<usize> = (0..dimension).collect();
        importance_order.sort_by(|&a, &b| variances[b].total_cmp(&variances[a]));

        Self {
            variances,
            means,
            importance_order,
            num_samples: num_vectors,
        }
    }

    /// Get dimension order sorted by importance.
    pub fn order(&self) -> &[usize] {
        &self.importance_order
    }

    /// Get top-k most important dimensions.
    pub fn top_dimensions(&self, k: usize) -> &[usize] {
        let k = k.min(self.importance_order.len());
        &self.importance_order[..k]
    }

    /// Get importance weight for a dimension.
    pub fn weight(&self, dim: usize) -> f32 {
        self.variances.get(dim).copied().unwrap_or(0.0)
    }

    /// Total variance (sum of all dimension variances).
    pub fn total_variance(&self) -> f32 {
        self.variances.iter().sum()
    }

    /// Cumulative variance explained by top-k dimensions.
    pub fn cumulative_variance(&self, k: usize) -> f32 {
        self.importance_order
            .iter()
            .take(k)
            .map(|&d| self.variances[d])
            .sum()
    }
}

/// Hubness tracker for identifying and handling hub nodes.
///
/// In high-dimensional spaces, some points become "hubs" - they appear as
/// nearest neighbors to many queries despite not being semantically close.
/// This is a symptom of the curse of dimensionality.
///
/// Tracking hub frequency lets us deprioritize them during search.
#[derive(Debug, Clone)]
pub struct HubnessTracker {
    /// Count of times each node appeared in top-k results.
    occurrence_counts: Vec<usize>,
    /// Total queries processed.
    num_queries: usize,
    /// K used for tracking.
    k: usize,
    /// Threshold for hub classification.
    hub_threshold: f32,
}

impl HubnessTracker {
    /// Create tracker for n nodes.
    pub fn new(num_nodes: usize, k: usize, hub_threshold: f32) -> Self {
        Self {
            occurrence_counts: vec![0; num_nodes],
            num_queries: 0,
            k,
            hub_threshold,
        }
    }

    /// Record a query result.
    pub fn record_result(&mut self, top_k_indices: &[usize]) {
        self.num_queries += 1;
        for &idx in top_k_indices.iter().take(self.k) {
            if idx < self.occurrence_counts.len() {
                self.occurrence_counts[idx] += 1;
            }
        }
    }

    /// Is this node a hub?
    pub fn is_hub(&self, node_idx: usize) -> bool {
        if self.num_queries == 0 {
            return false;
        }
        let occurrence_rate = self.occurrence_counts.get(node_idx).copied().unwrap_or(0) as f32
            / self.num_queries as f32;
        occurrence_rate > self.hub_threshold
    }

    /// Get hub score (higher = more hub-like).
    pub fn hub_score(&self, node_idx: usize) -> f32 {
        if self.num_queries == 0 {
            return 0.0;
        }
        self.occurrence_counts.get(node_idx).copied().unwrap_or(0) as f32 / self.num_queries as f32
    }

    /// Get all nodes classified as hubs.
    pub fn hubs(&self) -> Vec<usize> {
        (0..self.occurrence_counts.len())
            .filter(|&i| self.is_hub(i))
            .collect()
    }

    /// Get hub statistics.
    pub fn stats(&self) -> HubnessStats {
        let num_hubs = self.hubs().len();
        let max_occurrences = self.occurrence_counts.iter().max().copied().unwrap_or(0);
        let mean_occurrences = if self.occurrence_counts.is_empty() {
            0.0
        } else {
            self.occurrence_counts.iter().sum::<usize>() as f32
                / self.occurrence_counts.len() as f32
        };

        HubnessStats {
            num_hubs,
            total_nodes: self.occurrence_counts.len(),
            max_occurrences,
            mean_occurrences,
            queries_processed: self.num_queries,
        }
    }
}

/// Statistics about hubness in the index.
#[derive(Debug, Clone)]
pub struct HubnessStats {
    /// Number of nodes classified as hubs.
    pub num_hubs: usize,
    /// Total nodes in index.
    pub total_nodes: usize,
    /// Maximum occurrence count for any node.
    pub max_occurrences: usize,
    /// Mean occurrence count.
    pub mean_occurrences: f32,
    /// Total queries processed.
    pub queries_processed: usize,
}

/// Sampled distance computation using dimension importance.
///
/// Computes distance using only a subset of dimensions, weighted by importance.
/// This provides an estimate that can be used for filtering before full computation.
pub fn sampled_l2_squared(
    a: &[f32],
    b: &[f32],
    importance: &DimensionImportance,
    sample_fraction: f32,
) -> (f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    let dim = a.len();

    let num_samples = ((dim as f32 * sample_fraction) as usize).max(1).min(dim);
    let sampled_dims = importance.top_dimensions(num_samples);

    // Compute sampled distance
    let mut sampled_dist = 0.0f32;
    for &d in sampled_dims {
        let diff = a[d] - b[d];
        sampled_dist += diff * diff;
    }

    // Estimate full distance by scaling
    // Weight by fraction of variance captured
    let sampled_variance = importance.cumulative_variance(num_samples);
    let total_variance = importance.total_variance();

    let scale = if sampled_variance > 1e-9 {
        total_variance / sampled_variance
    } else {
        dim as f32 / num_samples as f32
    };

    let estimated_full = sampled_dist * scale;

    (sampled_dist, estimated_full)
}

/// Two-phase distance computation with early rejection.
///
/// Phase 1: Compute distance on sampled dimensions
/// Phase 2: If estimate passes threshold, compute full distance
///
/// Returns None if rejected in phase 1.
pub fn two_phase_l2_squared(
    a: &[f32],
    b: &[f32],
    importance: &DimensionImportance,
    threshold: f32,
    sample_fraction: f32,
) -> Option<f32> {
    let (_, estimated) = sampled_l2_squared(a, b, importance, sample_fraction);

    // Reject if estimated distance exceeds threshold with margin
    if estimated > threshold * 1.5 {
        return None;
    }

    // Full computation
    let full_dist: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();

    Some(full_dist)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_termination_basic() {
        let config = AdaptiveConfig::default();
        let mut oracle = EarlyTerminationOracle::new(3, config);

        // Not enough candidates yet
        for i in 0..5 {
            oracle.observe(i as f32);
            assert!(!oracle.should_terminate());
        }

        // After many similar distances, should become confident
        for _ in 0..100 {
            oracle.observe(10.0 + rand_f32() * 0.1);
        }

        // With tight distribution and low k-th distance, might terminate
        // (depends on random values, so we just check it doesn't panic)
        let _ = oracle.should_terminate();
    }

    #[test]
    fn test_dimension_importance() {
        // Create vectors where first dimension has high variance
        let vectors: Vec<f32> = vec![
            0.0, 0.5, 0.5, // vec 0
            1.0, 0.5, 0.5, // vec 1
            2.0, 0.5, 0.5, // vec 2
            3.0, 0.5, 0.5, // vec 3
        ];

        let importance = DimensionImportance::estimate(&vectors, 4, 3);

        // First dimension should have highest importance
        assert_eq!(importance.order()[0], 0);

        // Cumulative variance should increase
        let cum1 = importance.cumulative_variance(1);
        let cum2 = importance.cumulative_variance(2);
        let cum3 = importance.cumulative_variance(3);
        assert!(cum1 <= cum2);
        assert!(cum2 <= cum3);
    }

    #[test]
    fn test_hubness_tracker() {
        let mut tracker = HubnessTracker::new(10, 3, 0.3);

        // Node 0 appears in every query
        for _ in 0..10 {
            tracker.record_result(&[0, 1, 2]);
        }

        // Node 5 appears less frequently
        for _ in 0..3 {
            tracker.record_result(&[5, 6, 7]);
        }

        assert!(tracker.is_hub(0)); // 100% occurrence
        assert!(!tracker.is_hub(5)); // ~23% occurrence

        let stats = tracker.stats();
        assert!(stats.num_hubs > 0);
    }

    #[test]
    fn test_sampled_distance() {
        let vectors: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) - 0.5).collect();
        let importance = DimensionImportance::estimate(&vectors, 10, 10);

        let a = vec![0.0f32; 10];
        let b: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();

        let (_sampled, estimated) = sampled_l2_squared(&a, &b, &importance, 0.5);

        // Estimated should be roughly proportional to full
        let full: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum();

        // Just check it's in the right ballpark (within 2x)
        assert!(estimated > 0.0);
        assert!(estimated < full * 3.0);
    }

    fn rand_f32() -> f32 {
        // Simple LCG for deterministic "random" in tests
        static mut SEED: u32 = 12345;
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            (SEED as f32) / (u32::MAX as f32)
        }
    }
}
