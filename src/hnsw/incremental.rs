//! Incremental learning patterns for graph-based ANN indices.
//!
//! # The Temporal Locality Insight
//!
//! Real-world workloads exhibit temporal locality:
//! - Recently inserted vectors are often queried soon after
//! - Certain regions get "hot" during specific time periods
//! - Query patterns drift over time
//!
//! Traditional HNSW ignores this, treating all edges equally. But we can
//! exploit these patterns to improve search efficiency.
//!
//! # Research Context
//!
//! Several papers explore adaptive graph structures:
//!
//! | Paper | Technique | Key Insight |
//! |-------|-----------|-------------|
//! | **EnhanceGraph** (2025) | Edge refinement | Use search logs to strengthen useful edges |
//! | **Delta-EMG** (2025) | Monotonic updates | Guarantee search quality during updates |
//! | **FreshDiskANN** (2024) | Streaming merge | Efficient handling of insert/delete streams |
//! | **IP-DiskANN** (2025) | In-neighbor tracking | O(1) deletion via reverse edge lists |
//!
//! This module implements the general patterns, with specific algorithms
//! (IP-DiskANN, MN-RU) in separate modules.
//!
//! # Edge Refinement
//!
//! The core idea: **strengthen frequently traversed edges**.
//!
//! During search, we record which edges were useful (led to improved candidates).
//! Periodically, we:
//! 1. Analyze edge usage statistics
//! 2. Add edges between frequently co-visited nodes
//! 3. Remove underutilized edges (optional)
//!
//! This adapts the graph structure to actual query patterns.

use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for incremental learning.
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Enable edge usage tracking.
    pub track_edge_usage: bool,
    /// Window size for temporal statistics (number of queries).
    pub stats_window: usize,
    /// Minimum edge traversals before considering refinement.
    pub min_traversals: usize,
    /// Threshold for adding new edges (co-occurrence rate).
    pub edge_add_threshold: f32,
    /// Threshold for removing edges (usage rate).
    pub edge_remove_threshold: f32,
    /// Maximum new edges to add per refinement pass.
    pub max_edges_per_pass: usize,
    /// Whether to run refinement asynchronously.
    pub async_refinement: bool,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            track_edge_usage: true,
            stats_window: 10_000,
            min_traversals: 100,
            edge_add_threshold: 0.3,
            edge_remove_threshold: 0.01,
            max_edges_per_pass: 1000,
            async_refinement: true,
        }
    }
}

/// Statistics about edge usage during search.
#[derive(Debug, Default)]
pub struct EdgeStats {
    /// Number of times each edge was traversed.
    edge_traversals: HashMap<(usize, usize), u64>,
    /// Number of times traversal led to improvement.
    edge_improvements: HashMap<(usize, usize), u64>,
    /// Co-occurrence: pairs of nodes visited in same search.
    cooccurrence: HashMap<(usize, usize), u64>,
    /// Total searches tracked.
    total_searches: u64,
    /// Recent query entry points (for temporal locality).
    recent_entry_points: VecDeque<usize>,
}

impl EdgeStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an edge traversal during search.
    pub fn record_traversal(&mut self, from: usize, to: usize, improved: bool) {
        let key = (from.min(to), from.max(to));
        *self.edge_traversals.entry(key).or_default() += 1;
        if improved {
            *self.edge_improvements.entry(key).or_default() += 1;
        }
    }

    /// Record nodes visited together in a search.
    pub fn record_covisit(&mut self, nodes: &[usize]) {
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let key = (nodes[i].min(nodes[j]), nodes[i].max(nodes[j]));
                *self.cooccurrence.entry(key).or_default() += 1;
            }
        }
    }

    /// Record a completed search.
    pub fn record_search(&mut self, entry_point: usize) {
        self.total_searches += 1;
        self.recent_entry_points.push_back(entry_point);
        if self.recent_entry_points.len() > 1000 {
            self.recent_entry_points.pop_front();
        }
    }

    /// Get edge improvement rate.
    pub fn improvement_rate(&self, from: usize, to: usize) -> f32 {
        let key = (from.min(to), from.max(to));
        let traversals = self.edge_traversals.get(&key).copied().unwrap_or(0);
        let improvements = self.edge_improvements.get(&key).copied().unwrap_or(0);

        if traversals == 0 {
            return 0.0;
        }
        improvements as f32 / traversals as f32
    }

    /// Get co-occurrence rate for a pair of nodes.
    pub fn cooccurrence_rate(&self, a: usize, b: usize) -> f32 {
        if self.total_searches == 0 {
            return 0.0;
        }
        let key = (a.min(b), a.max(b));
        let count = self.cooccurrence.get(&key).copied().unwrap_or(0);
        count as f32 / self.total_searches as f32
    }

    /// Get most frequently co-visited pairs.
    pub fn top_cooccurrences(&self, n: usize) -> Vec<((usize, usize), f32)> {
        if self.total_searches == 0 {
            return Vec::new();
        }

        let mut pairs: Vec<_> = self
            .cooccurrence
            .iter()
            .map(|(&k, &v)| (k, v as f32 / self.total_searches as f32))
            .collect();

        pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
        pairs.truncate(n);
        pairs
    }

    /// Get underutilized edges.
    pub fn underutilized_edges(&self, threshold: f32) -> Vec<(usize, usize)> {
        self.edge_traversals
            .iter()
            .filter(|&(&_edge, &count)| {
                let rate = count as f32 / self.total_searches.max(1) as f32;
                rate < threshold
            })
            .map(|(&edge, _)| edge)
            .collect()
    }

    /// Clear statistics (call after refinement).
    pub fn clear(&mut self) {
        self.edge_traversals.clear();
        self.edge_improvements.clear();
        self.cooccurrence.clear();
        self.total_searches = 0;
    }

    /// Get total searches tracked.
    pub fn total_searches(&self) -> u64 {
        self.total_searches
    }
}

/// Suggestions for graph refinement based on usage patterns.
#[derive(Debug, Clone)]
pub struct RefinementSuggestions {
    /// Edges to add (high co-occurrence, no current edge).
    pub edges_to_add: Vec<(usize, usize, f32)>,
    /// Edges to remove (low utilization).
    pub edges_to_remove: Vec<(usize, usize)>,
    /// Nodes that should be entry points (frequently reached early).
    pub hot_entry_points: Vec<usize>,
}

/// Analyzer that produces refinement suggestions from statistics.
pub struct RefinementAnalyzer {
    config: IncrementalConfig,
}

impl RefinementAnalyzer {
    /// Create analyzer with config.
    pub fn new(config: IncrementalConfig) -> Self {
        Self { config }
    }

    /// Analyze statistics and produce refinement suggestions.
    pub fn analyze(
        &self,
        stats: &EdgeStats,
        existing_edges: &HashSet<(usize, usize)>,
    ) -> RefinementSuggestions {
        let mut edges_to_add = Vec::new();
        let mut edges_to_remove = Vec::new();

        // Find edges to add: high co-occurrence, not currently connected
        let top_cooc = stats.top_cooccurrences(self.config.max_edges_per_pass * 2);
        for ((a, b), rate) in top_cooc {
            if rate >= self.config.edge_add_threshold && !existing_edges.contains(&(a, b)) {
                edges_to_add.push((a, b, rate));
                if edges_to_add.len() >= self.config.max_edges_per_pass {
                    break;
                }
            }
        }

        // Find edges to remove: low utilization
        if stats.total_searches() >= self.config.min_traversals as u64 {
            edges_to_remove = stats.underutilized_edges(self.config.edge_remove_threshold);
        }

        // Find hot entry points
        let mut entry_counts: HashMap<usize, usize> = HashMap::new();
        for &ep in &stats.recent_entry_points {
            *entry_counts.entry(ep).or_default() += 1;
        }
        let mut hot_entry_points: Vec<_> = entry_counts.into_iter().collect();
        hot_entry_points.sort_by(|a, b| b.1.cmp(&a.1));
        let hot_entry_points: Vec<_> = hot_entry_points
            .into_iter()
            .take(10)
            .map(|(node, _)| node)
            .collect();

        RefinementSuggestions {
            edges_to_add,
            edges_to_remove,
            hot_entry_points,
        }
    }
}

/// Recency-weighted distance for exploiting temporal locality.
///
/// Recent insertions get a distance "bonus" so they're more likely to be
/// connected to incoming queries. This helps with the "freshness" problem
/// where newly inserted vectors are hard to find.
#[derive(Debug)]
pub struct RecencyWeighting {
    /// Insertion timestamps (or sequence numbers) per node.
    insertion_times: Vec<u64>,
    /// Current timestamp.
    current_time: u64,
    /// Decay factor (higher = faster decay of recency bonus).
    decay: f32,
    /// Maximum recency bonus (as fraction of distance).
    max_bonus: f32,
}

impl RecencyWeighting {
    /// Create new recency weighting.
    pub fn new(initial_capacity: usize, decay: f32, max_bonus: f32) -> Self {
        Self {
            insertion_times: vec![0; initial_capacity],
            current_time: 0,
            decay,
            max_bonus,
        }
    }

    /// Record insertion of a node.
    pub fn record_insertion(&mut self, node: usize) {
        let time = self.current_time;
        self.current_time += 1;
        if node >= self.insertion_times.len() {
            self.insertion_times.resize(node + 1, 0);
        }
        self.insertion_times[node] = time;
    }

    /// Get recency bonus for a node (0.0 to max_bonus).
    pub fn recency_bonus(&self, node: usize) -> f32 {
        let current = self.current_time;
        let inserted = self.insertion_times.get(node).copied().unwrap_or(0);

        if current <= inserted {
            return self.max_bonus;
        }

        let age = (current - inserted) as f32;
        self.max_bonus * (-self.decay * age).exp()
    }

    /// Adjust distance with recency bonus.
    ///
    /// Lower distance = closer, so we subtract the bonus.
    pub fn adjust_distance(&self, distance: f32, node: usize) -> f32 {
        let bonus = self.recency_bonus(node);
        (distance * (1.0 - bonus)).max(0.0)
    }
}

/// Tracker for query distribution drift.
///
/// Monitors how query patterns change over time, enabling proactive
/// index adaptation.
#[derive(Debug)]
pub struct DriftTracker {
    /// Running centroid of recent queries.
    query_centroid: Vec<f32>,
    /// Historical centroids (for drift detection).
    historical_centroids: VecDeque<Vec<f32>>,
    /// Number of queries in current window.
    window_count: usize,
    /// Window size.
    window_size: usize,
    /// Dimension.
    dimension: usize,
}

impl DriftTracker {
    /// Create new drift tracker.
    pub fn new(dimension: usize, window_size: usize) -> Self {
        Self {
            query_centroid: vec![0.0; dimension],
            historical_centroids: VecDeque::new(),
            window_count: 0,
            window_size,
            dimension,
        }
    }

    /// Record a query vector.
    pub fn record_query(&mut self, query: &[f32]) {
        if query.len() != self.dimension {
            return;
        }

        // Update running centroid
        self.window_count += 1;
        let alpha = 1.0 / self.window_count as f32;
        for (c, &q) in self.query_centroid.iter_mut().zip(query.iter()) {
            *c = (1.0 - alpha) * *c + alpha * q;
        }

        // Check if window is complete
        if self.window_count >= self.window_size {
            self.historical_centroids
                .push_back(self.query_centroid.clone());
            if self.historical_centroids.len() > 10 {
                self.historical_centroids.pop_front();
            }
            self.query_centroid = vec![0.0; self.dimension];
            self.window_count = 0;
        }
    }

    /// Measure drift from historical average.
    pub fn drift_magnitude(&self) -> f32 {
        if self.historical_centroids.len() < 2 {
            return 0.0;
        }

        // Compare current centroid to historical average
        let historical_avg: Vec<f32> = (0..self.dimension)
            .map(|d| {
                self.historical_centroids.iter().map(|c| c[d]).sum::<f32>()
                    / self.historical_centroids.len() as f32
            })
            .collect();

        // L2 distance from current to historical
        self.query_centroid
            .iter()
            .zip(historical_avg.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Is significant drift detected?
    pub fn has_drift(&self, threshold: f32) -> bool {
        self.drift_magnitude() > threshold
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_stats_basic() {
        let mut stats = EdgeStats::new();

        // Record some traversals
        stats.record_traversal(0, 1, true);
        stats.record_traversal(0, 1, false);
        stats.record_traversal(0, 1, true);

        assert!((stats.improvement_rate(0, 1) - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_cooccurrence() {
        let mut stats = EdgeStats::new();

        stats.record_covisit(&[0, 1, 2]);
        stats.record_search(0);
        stats.record_covisit(&[0, 1, 3]);
        stats.record_search(0);

        // 0,1 appeared together twice
        assert_eq!(stats.cooccurrence_rate(0, 1), 1.0);
        // 0,2 appeared together once
        assert_eq!(stats.cooccurrence_rate(0, 2), 0.5);
    }

    #[test]
    fn test_recency_weighting() {
        let mut rw = RecencyWeighting::new(10, 0.1, 0.2);

        rw.record_insertion(0);
        rw.record_insertion(1);
        rw.record_insertion(2);

        // Most recent insertion should have highest bonus
        let bonus_0 = rw.recency_bonus(0);
        let bonus_2 = rw.recency_bonus(2);
        assert!(bonus_2 > bonus_0);

        // Adjusted distance should be lower for recent nodes
        let dist = 1.0;
        let adj_0 = rw.adjust_distance(dist, 0);
        let adj_2 = rw.adjust_distance(dist, 2);
        assert!(adj_2 < adj_0);
    }

    #[test]
    fn test_refinement_analyzer() {
        let config = IncrementalConfig::default();
        let analyzer = RefinementAnalyzer::new(config);

        let mut stats = EdgeStats::new();
        // Create high co-occurrence between 0 and 5
        for _ in 0..100 {
            stats.record_covisit(&[0, 5]);
            stats.record_search(0);
        }

        let existing: HashSet<_> = [(0, 1), (1, 2)].into_iter().collect();
        let suggestions = analyzer.analyze(&stats, &existing);

        // Should suggest adding edge between 0 and 5
        assert!(suggestions
            .edges_to_add
            .iter()
            .any(|&(a, b, _)| { (a == 0 && b == 5) || (a == 5 && b == 0) }));
    }

    #[test]
    fn test_drift_tracker() {
        let mut tracker = DriftTracker::new(3, 10);

        // Record queries centered around origin
        for _ in 0..10 {
            tracker.record_query(&[0.0, 0.0, 0.0]);
        }

        // Record queries that have drifted
        for _ in 0..10 {
            tracker.record_query(&[1.0, 1.0, 1.0]);
        }

        assert!(tracker.drift_magnitude() > 0.0);
    }
}
