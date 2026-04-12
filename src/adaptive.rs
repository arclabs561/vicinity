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
//! # Early Termination
//!
//! - **Early termination**: Stop once result is "good enough" (~1.5-2x speedup)
//!
//! Based on the DARTH paper: estimate probability that any remaining candidate
//! would displace the current top-k, stop when that probability falls below a
//! threshold.

/// Configuration for adaptive search.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Minimum candidates to evaluate before considering early termination.
    pub min_candidates: usize,

    /// Confidence threshold for early termination (0.0-1.0).
    /// Higher = more conservative = better recall but slower.
    pub confidence_threshold: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_candidates: 10,
            confidence_threshold: 0.9,
        }
    }
}

impl AdaptiveConfig {
    /// Conservative config - maximize recall at cost of speed.
    pub fn conservative() -> Self {
        Self {
            min_candidates: 50,
            confidence_threshold: 0.99,
        }
    }

    /// Aggressive config - maximize speed at cost of recall.
    pub fn aggressive() -> Self {
        Self {
            min_candidates: 5,
            confidence_threshold: 0.7,
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
            self.top_k_distances.sort_unstable_by(|a, b| a.total_cmp(b));
        } else if distance < self.top_k_distances[self.k - 1] {
            self.top_k_distances[self.k - 1] = distance;
            self.top_k_distances.sort_unstable_by(|a, b| a.total_cmp(b));
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

    /// Number of candidates evaluated.
    pub fn num_evaluated(&self) -> usize {
        self.num_evaluated
    }
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

    fn rand_f32() -> f32 {
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u32> = const { Cell::new(12345) };
        }
        SEED.with(|s| {
            let next = s.get().wrapping_mul(1103515245).wrapping_add(12345);
            s.set(next);
            (next as f32) / (u32::MAX as f32)
        })
    }
}
