//! Probabilistic Edge Ordering (PEOs) for HNSW
//!
//! Implements probabilistic routing: test a subset of candidate edges before paying
//! the full cost of distance computations.
//!
//! Some work reports throughput improvements with minimal recall loss for graph-based ANN
//! by learning/estimating which edges are unlikely to improve the current best candidates.
//! Exact gains are dataset- and parameter-dependent.
//!
//! References:
//! - Lu, Xiao, Ishikawa (2024). "Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search." `https://arxiv.org/abs/2402.11354`
//!
//! Key techniques:
//! - Edge probability estimation based on angle/distance
//! - Early termination when probability of finding better neighbors is low
//! - Adaptive probability thresholds based on current search state

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Configuration for probabilistic routing.
#[derive(Debug, Clone)]
pub struct ProbabilisticRoutingConfig {
    /// Probability threshold for edge testing (higher = more aggressive pruning).
    pub probability_threshold: f32,
    /// Delta parameter for routing guarantee.
    pub delta: f32,
    /// Epsilon parameter for success probability.
    pub epsilon: f32,
    /// Whether to use adaptive thresholds.
    pub adaptive: bool,
    /// Minimum edges to always test.
    pub min_edges_to_test: usize,
    /// Maximum skip ratio (1 - fraction of edges tested).
    pub max_skip_ratio: f32,
}

impl Default for ProbabilisticRoutingConfig {
    fn default() -> Self {
        Self {
            probability_threshold: 0.3,
            delta: 1.0,
            epsilon: 0.1,
            adaptive: true,
            min_edges_to_test: 2,
            max_skip_ratio: 0.5,
        }
    }
}

impl ProbabilisticRoutingConfig {
    /// Aggressive pruning for maximum QPS.
    pub fn fast() -> Self {
        Self {
            probability_threshold: 0.5,
            delta: 1.2,
            epsilon: 0.15,
            adaptive: true,
            min_edges_to_test: 1,
            max_skip_ratio: 0.7,
        }
    }

    /// Conservative pruning for maximum recall.
    pub fn accurate() -> Self {
        Self {
            probability_threshold: 0.1,
            delta: 1.0,
            epsilon: 0.05,
            adaptive: true,
            min_edges_to_test: 4,
            max_skip_ratio: 0.3,
        }
    }
}

/// Edge probability estimator.
#[derive(Debug)]
pub struct EdgeProbabilityEstimator {
    /// Running estimate of local density.
    density_estimate: f32,
    /// Number of samples used for estimation.
    sample_count: u32,
    /// Configuration.
    config: ProbabilisticRoutingConfig,
}

impl EdgeProbabilityEstimator {
    /// Create a new estimator.
    pub fn new(config: ProbabilisticRoutingConfig) -> Self {
        Self {
            density_estimate: 1.0,
            sample_count: 0,
            config,
        }
    }

    /// Estimate probability that an edge leads to a better neighbor.
    ///
    /// Based on the angle between:
    /// - Vector from query to current best
    /// - Vector from current best to candidate neighbor
    ///
    /// Higher probability when angle is acute (moving toward query).
    pub fn estimate_edge_probability(
        &self,
        query: &[f32],
        current_best_dist: f32,
        current_pos: &[f32],
        neighbor_pos: &[f32],
    ) -> f32 {
        // Direction from current to query
        let to_query: Vec<f32> = query
            .iter()
            .zip(current_pos.iter())
            .map(|(q, c)| q - c)
            .collect();

        // Direction from current to neighbor
        let to_neighbor: Vec<f32> = neighbor_pos
            .iter()
            .zip(current_pos.iter())
            .map(|(n, c)| n - c)
            .collect();

        // Compute cosine of angle between directions
        let dot: f32 = crate::simd::dot(&to_query, &to_neighbor);
        let norm_q: f32 = crate::simd::norm(&to_query);
        let norm_n: f32 = crate::simd::norm(&to_neighbor);

        let cos_angle = if norm_q > 1e-10 && norm_n > 1e-10 {
            (dot / (norm_q * norm_n)).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Distance ratio: how far is neighbor relative to current best distance
        let neighbor_dist_from_current = norm_n;
        let dist_ratio = if current_best_dist > 1e-10 {
            (neighbor_dist_from_current / current_best_dist).min(2.0)
        } else {
            1.0
        };

        // Higher probability when:
        // 1. cos_angle is positive (moving toward query)
        // 2. dist_ratio is reasonable (not too far)
        let angle_factor = (cos_angle + 1.0) / 2.0; // Map [-1, 1] to [0, 1]
        let dist_factor = 1.0 / (1.0 + dist_ratio);

        // Combine factors with density adjustment
        let base_prob = angle_factor * dist_factor;
        let adjusted_prob = base_prob * self.density_estimate;

        adjusted_prob.clamp(0.0, 1.0)
    }

    /// Update density estimate based on observed search behavior.
    pub fn update_density(&mut self, neighbors_improved: usize, neighbors_tested: usize) {
        if neighbors_tested > 0 {
            let improvement_rate = neighbors_improved as f32 / neighbors_tested as f32;
            // Exponential moving average
            let alpha = 0.1;
            self.density_estimate =
                (1.0 - alpha) * self.density_estimate + alpha * improvement_rate;
            self.sample_count += 1;
        }
    }

    /// Get current probability threshold.
    pub fn get_threshold(&self) -> f32 {
        if self.config.adaptive && self.sample_count > 10 {
            // Adapt threshold based on observed density
            (self.config.probability_threshold * self.density_estimate).clamp(0.05, 0.8)
        } else {
            self.config.probability_threshold
        }
    }
}

/// Search candidate with probability info.
#[derive(Debug, Clone)]
struct ProbabilisticCandidate {
    id: u32,
    distance: f32,
    probability: f32,
}

impl PartialEq for ProbabilisticCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for ProbabilisticCandidate {}

impl PartialOrd for ProbabilisticCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProbabilisticCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller distance = higher priority
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

/// Probabilistic routing layer for HNSW search.
#[derive(Debug)]
pub struct ProbabilisticRouter {
    /// Configuration.
    config: ProbabilisticRoutingConfig,
    /// Probability estimator.
    estimator: EdgeProbabilityEstimator,
    /// Statistics.
    stats: ProbabilisticStats,
}

/// Statistics for probabilistic routing.
#[derive(Debug, Default, Clone)]
pub struct ProbabilisticStats {
    /// Total edges considered.
    pub edges_considered: u64,
    /// Edges actually tested (distance computed).
    pub edges_tested: u64,
    /// Edges skipped due to low probability.
    pub edges_skipped: u64,
    /// Edges that would have improved result (for measuring skip quality).
    pub beneficial_skips: u64,
    /// Total searches.
    pub total_searches: u64,
}

impl ProbabilisticStats {
    /// Get the skip ratio (fraction of edges skipped).
    pub fn skip_ratio(&self) -> f32 {
        if self.edges_considered > 0 {
            self.edges_skipped as f32 / self.edges_considered as f32
        } else {
            0.0
        }
    }

    /// Get the QPS improvement factor estimate.
    pub fn estimated_qps_factor(&self) -> f32 {
        if self.edges_tested > 0 {
            self.edges_considered as f32 / self.edges_tested as f32
        } else {
            1.0
        }
    }
}

impl ProbabilisticRouter {
    /// Create a new probabilistic router.
    pub fn new(config: ProbabilisticRoutingConfig) -> Self {
        let estimator = EdgeProbabilityEstimator::new(config.clone());
        Self {
            config,
            estimator,
            stats: ProbabilisticStats::default(),
        }
    }

    /// Filter neighbors to test based on probability.
    ///
    /// Returns indices of neighbors that should be tested.
    pub fn filter_neighbors<'a>(
        &mut self,
        query: &[f32],
        current_pos: &[f32],
        current_best_dist: f32,
        neighbors: &'a [(u32, &[f32])], // (id, position)
    ) -> Vec<(u32, &'a [f32], f32)> {
        // (id, position, probability)
        let threshold = self.estimator.get_threshold();
        let mut candidates: Vec<(u32, &'a [f32], f32)> = Vec::with_capacity(neighbors.len());

        for &(id, pos) in neighbors {
            self.stats.edges_considered += 1;

            let prob = self.estimator.estimate_edge_probability(
                query,
                current_best_dist,
                current_pos,
                pos,
            );

            candidates.push((id, pos, prob));
        }

        // Sort by probability (highest first)
        candidates.sort_by(|a, b| b.2.total_cmp(&a.2));

        // Determine how many to test
        let total = candidates.len();
        let min_to_test = self.config.min_edges_to_test.min(total);
        let max_to_skip = ((total as f32) * self.config.max_skip_ratio) as usize;

        // Keep candidates above threshold, but respect min/max constraints
        let mut to_test = 0;
        for (i, (_, _, prob)) in candidates.iter().enumerate() {
            if *prob >= threshold || to_test < min_to_test {
                to_test = i + 1;
            } else if i >= total - max_to_skip {
                break;
            }
        }

        let to_test = to_test.max(min_to_test).min(total);

        self.stats.edges_tested += to_test as u64;
        self.stats.edges_skipped += (total - to_test) as u64;

        candidates.truncate(to_test);
        candidates
    }

    /// Perform probabilistic beam search.
    ///
    /// This is a search implementation that uses probabilistic routing internally.
    pub fn search(
        &mut self,
        query: &[f32],
        entry_point: u32,
        entry_pos: &[f32],
        get_neighbors: impl Fn(u32) -> Vec<(u32, Vec<f32>)>,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        self.stats.total_searches += 1;

        let mut visited: HashSet<u32> = HashSet::new();
        let mut candidates: BinaryHeap<ProbabilisticCandidate> = BinaryHeap::new();
        let mut results: BinaryHeap<ProbabilisticCandidate> = BinaryHeap::new();

        // Initialize with entry point
        let entry_dist = euclidean_distance(query, entry_pos);
        candidates.push(ProbabilisticCandidate {
            id: entry_point,
            distance: entry_dist,
            probability: 1.0,
        });
        results.push(ProbabilisticCandidate {
            id: entry_point,
            distance: -entry_dist, // Max-heap for results (want to keep smallest)
            probability: 1.0,
        });
        visited.insert(entry_point);

        let mut current_best_dist = entry_dist;
        let mut neighbors_improved = 0;
        let mut neighbors_tested = 0;

        while let Some(current) = candidates.pop() {
            // Early termination: if current is worse than worst in results
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > -worst.distance {
                        break;
                    }
                }
            }

            // Get neighbors
            let raw_neighbors = get_neighbors(current.id);

            // Get current position for probability estimation
            // (In real implementation, would cache this or get from index)
            let current_pos: Vec<f32> = raw_neighbors
                .first()
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| vec![0.0; query.len()]);

            // Filter by probability
            let neighbors_with_pos: Vec<_> = raw_neighbors
                .iter()
                .filter(|(id, _)| !visited.contains(id))
                .map(|(id, pos)| (*id, pos.as_slice()))
                .collect();

            let filtered =
                self.filter_neighbors(query, &current_pos, current_best_dist, &neighbors_with_pos);

            for (neighbor_id, neighbor_pos, _prob) in filtered {
                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id);
                neighbors_tested += 1;

                let dist = euclidean_distance(query, neighbor_pos);

                if dist < current_best_dist {
                    current_best_dist = dist;
                    neighbors_improved += 1;
                }

                // Add to candidates if promising
                let should_add = results.len() < ef
                    || dist < -results.peek().map(|r| r.distance).unwrap_or(f32::INFINITY);

                if should_add {
                    candidates.push(ProbabilisticCandidate {
                        id: neighbor_id,
                        distance: dist,
                        probability: 1.0,
                    });
                    results.push(ProbabilisticCandidate {
                        id: neighbor_id,
                        distance: -dist,
                        probability: 1.0,
                    });

                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Update estimator
        self.estimator
            .update_density(neighbors_improved, neighbors_tested);

        // Extract results
        let mut final_results: Vec<(u32, f32)> =
            results.into_iter().map(|c| (c.id, -c.distance)).collect();
        final_results.sort_by(|a, b| a.1.total_cmp(&b.1));
        final_results
    }

    /// Get statistics.
    pub fn stats(&self) -> &ProbabilisticStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = ProbabilisticStats::default();
    }
}

/// Simple edge selector that uses probability to order edges.
#[derive(Debug)]
pub struct ProbabilisticEdgeSelector {
    /// Configuration.
    config: ProbabilisticRoutingConfig,
}

impl ProbabilisticEdgeSelector {
    /// Create a new selector.
    pub fn new(config: ProbabilisticRoutingConfig) -> Self {
        Self { config }
    }

    /// Order edges by estimated utility.
    ///
    /// Returns edges sorted by probability of leading to improvement.
    pub fn order_edges(
        &self,
        query: &[f32],
        current_pos: &[f32],
        current_dist: f32,
        edges: &[(u32, Vec<f32>)],
    ) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32, f32)> = edges
            .iter()
            .map(|(id, pos)| {
                let prob = estimate_improvement_probability(query, current_pos, current_dist, pos);
                let dist = euclidean_distance(query, pos);
                (*id, dist, prob)
            })
            .collect();

        // Sort by probability (high) then distance (low)
        scored.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.total_cmp(&b.1))
        });

        scored.into_iter().map(|(id, dist, _)| (id, dist)).collect()
    }
}

/// Estimate probability that moving to neighbor improves search.
fn estimate_improvement_probability(
    query: &[f32],
    _current_pos: &[f32],
    current_dist: f32,
    neighbor_pos: &[f32],
) -> f32 {
    let neighbor_dist = euclidean_distance(query, neighbor_pos);

    // Simple heuristic: probability based on distance ratio
    if neighbor_dist < current_dist {
        // Will definitely improve
        1.0
    } else {
        // Probability decreases as neighbor gets further
        let ratio = current_dist / neighbor_dist.max(1e-10);
        ratio.powi(2).clamp(0.0, 1.0)
    }
}

use crate::distance::l2_distance as euclidean_distance;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| ((seed as f32 * 0.1 + i as f32) * 0.01).sin())
            .collect()
    }

    #[test]
    fn test_probability_estimation() {
        let config = ProbabilisticRoutingConfig::default();
        let estimator = EdgeProbabilityEstimator::new(config);

        let query = vec![1.0, 0.0, 0.0];
        let current = vec![0.5, 0.0, 0.0];
        let current_dist = euclidean_distance(&query, &current);

        // Neighbor closer to query (should have high probability)
        let neighbor_good = vec![0.8, 0.0, 0.0];
        let prob_good =
            estimator.estimate_edge_probability(&query, current_dist, &current, &neighbor_good);

        // Neighbor further from query (should have lower probability)
        let neighbor_bad = vec![0.2, 0.0, 0.0];
        let prob_bad =
            estimator.estimate_edge_probability(&query, current_dist, &current, &neighbor_bad);

        assert!(prob_good > prob_bad);
    }

    #[test]
    fn test_filter_neighbors() {
        let config = ProbabilisticRoutingConfig::default();
        let mut router = ProbabilisticRouter::new(config);

        let query = make_vector(64, 100);
        let current = make_vector(64, 50);
        let current_dist = euclidean_distance(&query, &current);

        // Create some neighbors
        let neighbors: Vec<(u32, Vec<f32>)> =
            (0..10).map(|i| (i, make_vector(64, i * 10))).collect();

        let neighbors_ref: Vec<_> = neighbors
            .iter()
            .map(|(id, pos)| (*id, pos.as_slice()))
            .collect();

        let filtered = router.filter_neighbors(&query, &current, current_dist, &neighbors_ref);

        // Should filter some neighbors
        assert!(filtered.len() <= neighbors.len());
        assert!(filtered.len() >= router.config.min_edges_to_test.min(neighbors.len()));
    }

    #[test]
    fn test_stats_tracking() {
        let config = ProbabilisticRoutingConfig::default();
        let mut router = ProbabilisticRouter::new(config);

        let query = make_vector(64, 100);
        let current = make_vector(64, 50);
        let current_dist = euclidean_distance(&query, &current);

        let neighbors: Vec<(u32, Vec<f32>)> =
            (0..20).map(|i| (i, make_vector(64, i * 5))).collect();

        let neighbors_ref: Vec<_> = neighbors
            .iter()
            .map(|(id, pos)| (*id, pos.as_slice()))
            .collect();

        router.filter_neighbors(&query, &current, current_dist, &neighbors_ref);

        assert!(router.stats.edges_considered > 0);
        assert!(router.stats.edges_tested > 0);
        // Should have some skip ratio
        let skip_ratio = router.stats.skip_ratio();
        assert!((0.0..=1.0).contains(&skip_ratio));
    }

    #[test]
    fn test_edge_selector() {
        let config = ProbabilisticRoutingConfig::default();
        let selector = ProbabilisticEdgeSelector::new(config);

        let query = vec![1.0, 0.0];
        let current = vec![0.5, 0.0];
        let current_dist = euclidean_distance(&query, &current);

        let edges = vec![
            (0, vec![0.2, 0.0]), // Further from query
            (1, vec![0.8, 0.0]), // Closer to query
            (2, vec![0.5, 0.5]), // Different direction
        ];

        let ordered = selector.order_edges(&query, &current, current_dist, &edges);

        // Edge 1 (closer to query) should be first
        assert_eq!(ordered[0].0, 1);
    }

    #[test]
    fn test_config_presets() {
        let fast = ProbabilisticRoutingConfig::fast();
        let accurate = ProbabilisticRoutingConfig::accurate();

        // Fast should have higher threshold (more aggressive)
        assert!(fast.probability_threshold > accurate.probability_threshold);
        // Accurate should test more edges
        assert!(accurate.min_edges_to_test > fast.min_edges_to_test);
    }
}
