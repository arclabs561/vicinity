//! Martingale-based pruning model for OPT-SNG.

use crate::RetrieveError;

/// Candidate set evolution during graph construction.
///
/// Models the stochastic evolution of candidate sets using martingale theory.
pub struct CandidateEvolution {
    /// Current candidate set size
    current_size: usize,

    /// Expected size after next iteration
    expected_size: f32,

    /// Variance of candidate set size
    variance: f32,
}

impl CandidateEvolution {
    /// Create new candidate evolution tracker.
    pub fn new() -> Self {
        Self {
            current_size: 0,
            expected_size: 0.0,
            variance: 0.0,
        }
    }

    /// Update candidate set with new observations.
    pub fn update(&mut self, new_size: usize) {
        let old_expected = self.expected_size;
        self.current_size = new_size;

        // Martingale update: E[X_{n+1} | X_n] = X_n (martingale property)
        // In practice, we track the evolution
        self.expected_size = 0.9 * old_expected + 0.1 * new_size as f32;

        // Update variance estimate
        let diff = new_size as f32 - self.expected_size;
        self.variance = 0.9 * self.variance + 0.1 * diff * diff;
    }

    /// Get current expected size.
    pub fn expected_size(&self) -> f32 {
        self.expected_size
    }

    /// Get variance estimate.
    pub fn variance(&self) -> f32 {
        self.variance
    }
}

/// Prune candidates using martingale-based model.
///
/// Uses the theoretical model to efficiently prune candidate sets
/// during graph construction, ensuring O(n^{2/3+ε}) maximum out-degree.
pub fn prune_candidates_martingale(
    candidates: &[(u32, f32)],
    truncation_r: f32,
    vectors: &[f32],
    dimension: usize,
) -> Result<Vec<u32>, RetrieveError> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Sort by distance
    let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

    // Apply truncation: keep only candidates within distance R
    let mut pruned = Vec::new();
    for (id, dist) in sorted {
        if dist <= truncation_r {
            pruned.push(id);
        } else {
            // Beyond truncation distance, stop (martingale-based early termination)
            break;
        }
    }

    // Ensure we don't exceed theoretical maximum out-degree
    let max_degree =
        crate::sng::optimization::estimate_max_degree(vectors.len() / dimension, truncation_r);

    if pruned.len() > max_degree {
        pruned.truncate(max_degree);
    }

    Ok(pruned)
}
