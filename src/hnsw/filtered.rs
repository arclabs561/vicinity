//! ACORN-style filtered HNSW search.
//!
//! Implements filter-aware traversal strategies for HNSW that maintain
//! navigability when filters eliminate nodes from consideration.
//!
//! # Algorithm Overview
//!
//! Standard HNSW with post-filtering fails when filters are restrictive
//! because the search may land in regions where few nodes match the filter.
//! ACORN solves this with:
//!
//! 1. **Two-hop expansion**: When visiting a node, examine both direct
//!    neighbors AND neighbors-of-neighbors (2-hop)
//! 2. **Filter-first**: Evaluate filter predicate before computing distances
//! 3. **Adaptive behavior**: Use standard traversal when dense matches exist,
//!    expand to 2-hop when sparse
//!
//! # References
//!
//! - Patel et al. (2024): "ACORN: Performant and Predicate-Agnostic Search
//!   Over Vector Embeddings and Structured Data"
//! - Weaviate blog: "Speed Up Filtered Vector Search"

use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet, VecDeque};

const MAX_VISITED_CAPACITY_HINT: usize = 1_000_000;

/// Filter predicate for ACORN search.
pub trait FilterPredicate: Sync {
    /// Check if a node passes the filter.
    fn matches(&self, node_id: u32) -> bool;
}

/// Simple function-based filter.
pub struct FnFilter<F: Fn(u32) -> bool + Sync>(pub F);

impl<F: Fn(u32) -> bool + Sync> FilterPredicate for FnFilter<F> {
    fn matches(&self, node_id: u32) -> bool {
        self.0(node_id)
    }
}

/// Adapter that bridges [`MetadataFilter`](crate::filtering::MetadataFilter) to [`FilterPredicate`].
///
/// Allows using the metadata filtering system with ACORN search.
///
/// ```rust,no_run
/// use vicinity::hnsw::filtered::MetadataFilterAdapter;
/// use vicinity::filtering::{MetadataFilter, MetadataStore};
///
/// let filter = MetadataFilter::equals("color", "red");
/// let store = MetadataStore::new();
/// let adapter = MetadataFilterAdapter::new(&filter, &store);
/// // `adapter` implements `FilterPredicate` and can be passed to `acorn_search`
/// ```
pub struct MetadataFilterAdapter<'a> {
    filter: &'a crate::filtering::MetadataFilter,
    store: &'a crate::filtering::MetadataStore,
}

impl<'a> MetadataFilterAdapter<'a> {
    /// Create a new adapter from a metadata filter and store.
    pub fn new(
        filter: &'a crate::filtering::MetadataFilter,
        store: &'a crate::filtering::MetadataStore,
    ) -> Self {
        Self { filter, store }
    }
}

impl FilterPredicate for MetadataFilterAdapter<'_> {
    fn matches(&self, doc_id: u32) -> bool {
        self.store.matches(doc_id, self.filter)
    }
}

/// Always-pass filter (no filtering).
pub struct NoFilter;

impl FilterPredicate for NoFilter {
    fn matches(&self, _node_id: u32) -> bool {
        true
    }
}

/// ACORN search configuration.
///
/// # Selectivity regimes
///
/// 2-hop expansion (Patel et al. SIGMOD 2024) is most effective when the
/// filter passes 2-20% of nodes. Outside that band:
///
/// - **>20% selectivity**: 2-hop is mostly overhead; standard HNSW search
///   with post-filter performs comparably. The branch still fires but
///   adds little.
/// - **<2% selectivity**: 2-hop hits a recall floor near 0.2 regardless
///   of `max_two_hop_neighbors` (Weaviate engineering data; ACORN paper
///   §5.4). The qualifying set is too sparse for graph traversal to
///   find enough hits even with aggressive expansion. At this regime,
///   prefer [`selectivity_search`] with `matching_ids` populated -- the
///   `Low` branch falls back to a brute-force scan over the pre-
///   filtered ID set, which is the right answer below ~2%. Curator
///   (arxiv:2601.01291) is the published alternative when pre-filter
///   IDs aren't available; ACORN simply isn't designed for this regime.
#[derive(Clone, Debug)]
pub struct AcornConfig {
    /// Enable two-hop expansion when filter is selective
    pub enable_two_hop: bool,
    /// Unused since two-hop became unconditional. Retained for API compatibility.
    pub two_hop_threshold: f32,
    /// Maximum two-hop neighbors to examine per node
    pub max_two_hop_neighbors: usize,
    /// Expansion factor for candidate pool
    pub ef_search: usize,
}

impl Default for AcornConfig {
    fn default() -> Self {
        Self {
            enable_two_hop: true,
            two_hop_threshold: 0.3, // Switch to 2-hop if <30% pass filter
            max_two_hop_neighbors: 32,
            ef_search: 100,
        }
    }
}

/// Search state tracking for adaptive behavior.
struct SearchState {
    visited: HashSet<u32>,
    filtered_count: usize,
    visited_count: usize,
}

impl SearchState {
    fn new(capacity_hint: usize) -> Self {
        Self {
            visited: HashSet::with_capacity(capacity_hint),
            filtered_count: 0,
            visited_count: 0,
        }
    }

    fn visit(&mut self, node_id: u32, passes_filter: bool) -> bool {
        if self.visited.insert(node_id) {
            self.visited_count += 1;
            if passes_filter {
                self.filtered_count += 1;
            }
            true
        } else {
            false
        }
    }

    fn filter_ratio(&self) -> f32 {
        if self.visited_count == 0 {
            1.0
        } else {
            self.filtered_count as f32 / self.visited_count as f32
        }
    }
}

fn visited_capacity_hint(config: &AcornConfig, k: usize) -> usize {
    config
        .ef_search
        .saturating_mul(3)
        .saturating_add(config.max_two_hop_neighbors)
        .saturating_add(k)
        .saturating_add(1)
        .min(MAX_VISITED_CAPACITY_HINT)
}

/// Candidate for search (ordered by distance, reversed for max-heap).
#[derive(Clone, Copy)]
struct Candidate {
    node_id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority (pops largest first).
        // The frontier stores negated distances so pop() returns the closest node.
        // The results heap stores positive distances so pop() evicts the farthest.
        // Use total_cmp for IEEE 754 total ordering (NaN-safe).
        self.distance.total_cmp(&other.distance)
    }
}

fn insert_bounded_result(results: &mut BinaryHeap<Candidate>, limit: usize, candidate: Candidate) {
    if limit == 0 {
        return;
    }
    if results.len() < limit {
        results.push(candidate);
        return;
    }
    if let Some(mut worst) = results.peek_mut() {
        if candidate.distance < worst.distance {
            *worst = candidate;
        }
    }
}

/// Internal counters for an ACORN search.
///
/// Useful for regression guards that want to assert which branches fired
/// (rather than only the final recall, which is a noisier proxy -- see
/// `test_design_pitfalls.md`). Returned from
/// [`acorn_search_with_stats`]; [`acorn_search`] discards it.
///
/// `#[non_exhaustive]`: new counters may be added without bumping major.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct AcornStats {
    /// Number of times the 2-hop expansion branch was entered, i.e. the
    /// search visited a non-matching neighbor while
    /// `AcornConfig::enable_two_hop` was set. Each entry triggers up to
    /// `max_two_hop_neighbors` 2-hop visits.
    pub two_hop_invocations: u64,
    /// Total number of distinct 2-hop nodes actually examined across all
    /// invocations. Bounded above by
    /// `two_hop_invocations * max_two_hop_neighbors`. Zero when the
    /// 2-hop branch never fired or every visited 2-hop node had been
    /// seen already.
    pub two_hop_nodes_examined: u64,
}

/// ACORN-style filtered search on HNSW graph.
///
/// This function performs filtered k-NN search with adaptive two-hop expansion.
///
/// # Arguments
/// * `query` - Query vector
/// * `k` - Number of results to return
/// * `config` - ACORN configuration
/// * `filter` - Filter predicate
/// * `get_neighbors` - Function to get neighbors of a node
/// * `compute_distance` - Function to compute distance between query and node
/// * `entry_point` - Starting node for search
///
/// # Returns
/// Vector of (node_id, distance) pairs for nodes passing the filter.
///
/// Use [`acorn_search_with_stats`] when you also need the internal
/// branch-fired counters (`AcornStats`) for testing.
pub fn acorn_search<F, N, D, G>(
    k: usize,
    config: &AcornConfig,
    filter: &F,
    get_neighbors: N,
    compute_distance: D,
    entry_point: u32,
) -> Result<Vec<(u32, f32)>, RetrieveError>
where
    F: FilterPredicate,
    N: Fn(u32) -> G,
    G: AsRef<[u32]>,
    D: Fn(u32) -> f32,
{
    acorn_search_with_stats(
        k,
        config,
        filter,
        get_neighbors,
        compute_distance,
        entry_point,
    )
    .map(|(results, _)| results)
}

/// Same as [`acorn_search`] but also returns the internal `AcornStats`
/// counters. Equivalent in behavior; only the return type differs.
pub fn acorn_search_with_stats<F, N, D, G>(
    k: usize,
    config: &AcornConfig,
    filter: &F,
    get_neighbors: N,
    compute_distance: D,
    entry_point: u32,
) -> Result<(Vec<(u32, f32)>, AcornStats), RetrieveError>
where
    F: FilterPredicate,
    N: Fn(u32) -> G,
    G: AsRef<[u32]>,
    D: Fn(u32) -> f32,
{
    let mut stats = AcornStats::default();
    let mut state = SearchState::new(visited_capacity_hint(config, k));

    // Result candidates (filtered nodes only)
    let mut results: BinaryHeap<Candidate> = BinaryHeap::new();

    // Search frontier
    let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();

    // Initialize with entry point
    let entry_passes = filter.matches(entry_point);
    state.visit(entry_point, entry_passes);
    let entry_dist = compute_distance(entry_point);

    frontier.push(Candidate {
        node_id: entry_point,
        distance: -entry_dist, // Negate for min-heap behavior
    });

    if entry_passes {
        results.push(Candidate {
            node_id: entry_point,
            distance: entry_dist,
        });
    }

    // Track worst result distance for pruning
    let mut worst_result_dist = f32::INFINITY;

    while let Some(current) = frontier.pop() {
        let current_dist = -current.distance; // Un-negate

        // Early termination: only if we have enough results AND frontier is worse
        // Be more conservative about stopping when filter is selective
        let can_stop = results.len() >= k && current_dist > worst_result_dist * 1.5; // 50% margin
        if can_stop && state.filter_ratio() > 0.3 {
            break;
        }

        // Get direct neighbors
        let neighbors = get_neighbors(current.node_id);

        // Process direct neighbors
        for &neighbor in neighbors.as_ref() {
            let neighbor_passes = filter.matches(neighbor);
            if !state.visit(neighbor, neighbor_passes) {
                continue; // Already visited
            }

            // Compute distance for all neighbors (needed for navigation)
            let dist = compute_distance(neighbor);

            if neighbor_passes {
                insert_bounded_result(
                    &mut results,
                    k,
                    Candidate {
                        node_id: neighbor,
                        distance: dist,
                    },
                );

                // Update worst distance
                if let Some(worst) = results.peek() {
                    worst_result_dist = worst.distance;
                }
            }

            // Add to frontier for exploration (even if doesn't pass filter)
            // This is critical: we need to navigate through non-matching nodes
            if dist < worst_result_dist * 2.0 || results.len() < k {
                frontier.push(Candidate {
                    node_id: neighbor,
                    distance: -dist,
                });
            }

            // ACORN-1: unconditional 2-hop expansion through non-matching nodes.
            // Per Kraft et al. SIGMOD 2024: every node that fails the predicate
            // gets its neighbors enqueued, preventing graph disconnection under
            // high selectivity.
            if config.enable_two_hop && !neighbor_passes {
                stats.two_hop_invocations += 1;
                let two_hop_neighbors = get_neighbors(neighbor);
                let mut two_hop_count = 0;

                for &two_hop in two_hop_neighbors.as_ref() {
                    if two_hop_count >= config.max_two_hop_neighbors {
                        break;
                    }

                    let two_hop_passes = filter.matches(two_hop);
                    if !state.visit(two_hop, two_hop_passes) {
                        continue;
                    }
                    stats.two_hop_nodes_examined += 1;

                    let two_hop_dist = compute_distance(two_hop);

                    if two_hop_passes {
                        insert_bounded_result(
                            &mut results,
                            k,
                            Candidate {
                                node_id: two_hop,
                                distance: two_hop_dist,
                            },
                        );

                        if let Some(worst) = results.peek() {
                            worst_result_dist = worst.distance;
                        }
                    }

                    // Also add two-hop to frontier
                    if two_hop_dist < worst_result_dist * 2.0 || results.len() < k {
                        frontier.push(Candidate {
                            node_id: two_hop,
                            distance: -two_hop_dist,
                        });
                    }

                    two_hop_count += 1;
                }
            }
        }

        // Standard HNSW termination: stop when frontier is exhausted or
        // we've explored enough candidates. Cap at ef * 3 (not ef * 10)
        // to avoid runaway exploration while allowing sufficient coverage.
        if results.len() >= k && state.visited_count >= config.ef_search * 3 {
            break;
        }
    }

    // Convert results to sorted vector
    let mut result_vec: Vec<(u32, f32)> = results
        .into_iter()
        .map(|c| (c.node_id, c.distance))
        .collect();

    result_vec.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    result_vec.truncate(k);

    Ok((result_vec, stats))
}

/// Selectivity regime for filtered search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectivityRegime {
    /// >10% of vectors match: standard ACORN 2-hop is sufficient.
    High,
    /// 1-10% match: aggressive over-retrieval + 2-hop.
    Medium,
    /// <1% match: pre-filtered brute-force over matching IDs.
    Low,
}

/// Configuration for selectivity-aware filtered search.
#[derive(Clone, Debug)]
pub struct SelectivityConfig {
    /// Threshold between high and medium selectivity (default: 0.10).
    pub high_threshold: f32,
    /// Threshold between medium and low selectivity (default: 0.01).
    pub low_threshold: f32,
    /// Number of probe nodes to estimate selectivity (default: 50).
    pub probe_size: usize,
    /// ef_search multiplier for medium selectivity (default: 4).
    pub medium_ef_multiplier: usize,
    /// Override the automatic regime detection.
    pub force_regime: Option<SelectivityRegime>,
}

impl Default for SelectivityConfig {
    fn default() -> Self {
        Self {
            high_threshold: 0.10,
            low_threshold: 0.01,
            probe_size: 50,
            medium_ef_multiplier: 4,
            force_regime: None,
        }
    }
}

/// Estimate selectivity by probing random neighbors from the entry point.
fn estimate_selectivity<F, N, G>(
    filter: &F,
    get_neighbors: &N,
    entry_point: u32,
    probe_size: usize,
) -> f32
where
    F: FilterPredicate,
    N: Fn(u32) -> G,
    G: AsRef<[u32]>,
{
    let mut visited = HashSet::new();
    let mut queue = VecDeque::with_capacity(probe_size);
    queue.push_back(entry_point);
    let mut matches = 0u32;
    let mut total = 0u32;

    while total < probe_size as u32 {
        let node = match queue.pop_front() {
            Some(n) => n,
            None => break,
        };
        if !visited.insert(node) {
            continue;
        }
        total += 1;
        if filter.matches(node) {
            matches += 1;
        }
        let neighbors = get_neighbors(node);
        for &neighbor in neighbors.as_ref() {
            if !visited.contains(&neighbor) {
                queue.push_back(neighbor);
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        matches as f32 / total as f32
    }
}

/// Selectivity-aware filtered search.
///
/// Automatically detects the selectivity regime and adapts the search strategy:
///
/// - **High selectivity** (>10% match): standard ACORN 2-hop
/// - **Medium selectivity** (1-10%): ACORN with aggressive over-retrieval
/// - **Low selectivity** (<1%): brute-force scan over pre-filtered candidates
///
/// For low selectivity, `matching_ids` must be provided -- the caller pre-computes
/// which IDs match the filter (e.g., from an inverted index on metadata).
///
/// # Arguments
/// * `k` - Number of results
/// * `config` - Selectivity configuration
/// * `filter` - Filter predicate
/// * `get_neighbors` - Graph neighbor function
/// * `compute_distance` - Distance function
/// * `entry_point` - HNSW entry point
/// * `matching_ids` - Pre-computed matching IDs for low-selectivity fallback.
///   If `None` and regime is Low, falls back to aggressive ACORN.
pub fn selectivity_search<F, N, D, G>(
    k: usize,
    config: &SelectivityConfig,
    filter: &F,
    get_neighbors: N,
    compute_distance: D,
    entry_point: u32,
    matching_ids: Option<&[u32]>,
) -> Result<Vec<(u32, f32)>, RetrieveError>
where
    F: FilterPredicate,
    N: Fn(u32) -> G,
    G: AsRef<[u32]>,
    D: Fn(u32) -> f32,
{
    // Determine regime
    let regime = config.force_regime.unwrap_or_else(|| {
        let selectivity =
            estimate_selectivity(filter, &get_neighbors, entry_point, config.probe_size);
        if selectivity >= config.high_threshold {
            SelectivityRegime::High
        } else if selectivity >= config.low_threshold {
            SelectivityRegime::Medium
        } else {
            SelectivityRegime::Low
        }
    });

    match regime {
        SelectivityRegime::High => {
            // Standard ACORN
            acorn_search(
                k,
                &AcornConfig::default(),
                filter,
                get_neighbors,
                compute_distance,
                entry_point,
            )
        }
        SelectivityRegime::Medium => {
            // ACORN with aggressive over-retrieval
            let acorn_config = AcornConfig {
                enable_two_hop: true,
                two_hop_threshold: 0.5, // More aggressive two-hop trigger
                max_two_hop_neighbors: 64,
                ef_search: AcornConfig::default().ef_search * config.medium_ef_multiplier,
            };
            acorn_search(
                k,
                &acorn_config,
                filter,
                get_neighbors,
                compute_distance,
                entry_point,
            )
        }
        SelectivityRegime::Low => {
            // Low selectivity: brute-force scan over matching IDs
            if let Some(ids) = matching_ids {
                let mut candidates: Vec<(u32, f32)> =
                    ids.iter().map(|&id| (id, compute_distance(id))).collect();
                candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                candidates.truncate(k);
                Ok(candidates)
            } else {
                // No pre-filtered IDs available, fall back to very aggressive ACORN
                let acorn_config = AcornConfig {
                    enable_two_hop: true,
                    two_hop_threshold: 0.8,
                    max_two_hop_neighbors: 128,
                    ef_search: AcornConfig::default().ef_search * config.medium_ef_multiplier * 2,
                };
                acorn_search(
                    k,
                    &acorn_config,
                    filter,
                    get_neighbors,
                    compute_distance,
                    entry_point,
                )
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn mock_graph() -> (Vec<Vec<u32>>, Vec<f32>) {
        // 10-node fully connected graph for reliable navigation
        // Each node connects to its neighbors within distance 2
        let neighbors = vec![
            vec![1, 2, 3],          // 0
            vec![0, 2, 3, 4],       // 1
            vec![0, 1, 3, 4, 5],    // 2
            vec![0, 1, 2, 4, 5, 6], // 3
            vec![1, 2, 3, 5, 6, 7], // 4
            vec![2, 3, 4, 6, 7, 8], // 5
            vec![3, 4, 5, 7, 8, 9], // 6
            vec![4, 5, 6, 8, 9],    // 7
            vec![5, 6, 7, 9],       // 8
            vec![6, 7, 8],          // 9
        ];

        // Distances from query - lower is better
        let distances = vec![0.5, 0.3, 0.6, 0.4, 0.7, 0.2, 0.8, 0.1, 0.9, 0.35];

        (neighbors, distances)
    }

    #[test]
    fn test_acorn_no_filter() {
        let (neighbors, distances) = mock_graph();

        let config = AcornConfig {
            enable_two_hop: true,
            two_hop_threshold: 0.3,
            max_two_hop_neighbors: 32,
            ef_search: 100,
        };

        let results = acorn_search(
            5, // Get more results to ensure we find best ones
            &config,
            &NoFilter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
        )
        .unwrap();

        assert!(!results.is_empty(), "Should find some results");
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1, "Results should be sorted");
        }
    }

    #[test]
    fn test_acorn_with_filter() {
        let (neighbors, distances) = mock_graph();

        // Filter: only even nodes (0, 2, 4, 6, 8)
        let filter = FnFilter(|id: u32| id.is_multiple_of(2));

        let results = acorn_search(
            3,
            &AcornConfig::default(),
            &filter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
        )
        .unwrap();

        // All results should be even
        for (id, _) in &results {
            assert_eq!(id % 2, 0, "Node {} should be even", id);
        }
    }

    #[test]
    fn test_selectivity_high_regime() {
        let (neighbors, distances) = mock_graph();

        // 50% pass: high selectivity
        let filter = FnFilter(|id: u32| id.is_multiple_of(2));

        let results = selectivity_search(
            3,
            &SelectivityConfig::default(),
            &filter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
            None,
        )
        .unwrap();

        assert!(!results.is_empty());
        for (id, _) in &results {
            assert_eq!(id % 2, 0);
        }
    }

    #[test]
    fn test_selectivity_low_with_prefiltered_ids() {
        let (_, distances) = mock_graph();

        // Only node 7 matches (0.1 distance)
        let filter = FnFilter(|id: u32| id == 7);
        let matching = vec![7u32];

        let results = selectivity_search(
            1,
            &SelectivityConfig {
                force_regime: Some(SelectivityRegime::Low),
                ..Default::default()
            },
            &filter,
            |_| vec![], // neighbors don't matter for brute-force
            |id| distances[id as usize],
            0,
            Some(&matching),
        )
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 7);
    }

    #[test]
    fn test_selectivity_forced_regime() {
        let (neighbors, distances) = mock_graph();
        let filter = FnFilter(|id: u32| id < 5);

        let results = selectivity_search(
            2,
            &SelectivityConfig {
                force_regime: Some(SelectivityRegime::Medium),
                ..Default::default()
            },
            &filter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
            None,
        )
        .unwrap();

        assert!(!results.is_empty());
        for (id, _) in &results {
            assert!(*id < 5);
        }
    }

    #[test]
    fn test_estimate_selectivity() {
        let (neighbors, _) = mock_graph();

        // 50% filter
        let filter = FnFilter(|id: u32| id.is_multiple_of(2));
        let sel = estimate_selectivity(&filter, &|id: u32| neighbors[id as usize].clone(), 0, 10);
        assert!(sel > 0.3 && sel < 0.7, "expected ~0.5, got {sel}");

        // 10% filter
        let filter = FnFilter(|id: u32| id == 0);
        let sel = estimate_selectivity(&filter, &|id: u32| neighbors[id as usize].clone(), 0, 10);
        assert!(sel < 0.2, "expected ~0.1, got {sel}");
    }

    #[test]
    fn test_acorn_selective_filter() {
        let (neighbors, distances) = mock_graph();

        // Filter: only nodes 8 or 9 (far end of graph)
        let filter = FnFilter(|id: u32| id >= 8);

        let config = AcornConfig {
            enable_two_hop: true,
            two_hop_threshold: 0.8, // High threshold to force two-hop early
            max_two_hop_neighbors: 32,
            ef_search: 100,
        };

        let results = acorn_search(
            2,
            &config,
            &filter,
            |id| neighbors[id as usize].clone(),
            |id| distances[id as usize],
            0,
        )
        .unwrap();

        // Should find nodes 8 and/or 9 through expansion
        assert!(!results.is_empty(), "Should find at least one node >= 8");
        for (id, _) in &results {
            assert!(*id >= 8, "Node {} should be >= 8", id);
        }
    }

    /// One-shot measurement of the wrapper-vs-direct cost for AcornStats.
    /// `acorn_search` is implemented as a thin wrapper around
    /// `acorn_search_with_stats` that discards the stats. In release
    /// builds with inlining enabled this should be free; this probe
    /// confirms it. If the gap is meaningful (>5%) the wrapper needs
    /// `#[inline(always)]` or the call sites should call
    /// `_with_stats` directly. Marked `#[ignore]` because it's a
    /// measurement, not a regression guard.
    ///
    /// Run with:
    ///   cargo test --release --features hnsw
    ///       hnsw::filtered::tests::acorn_stats_overhead_probe
    ///       -- --ignored --nocapture
    #[test]
    #[ignore = "measurement only; run with --release --ignored --nocapture"]
    fn acorn_stats_overhead_probe() {
        use std::time::Instant;

        let (neighbors, distances) = mock_graph();
        let filter = FnFilter(|id: u32| id.is_multiple_of(2));
        let config = AcornConfig::default();
        let iters = 100_000;

        // Warm up both code paths to prime icache + branch predictor; the
        // first measured loop otherwise pays a one-time setup cost that
        // shows up as artificial overhead for whichever runs first.
        for _ in 0..1000 {
            let _ = acorn_search(
                3,
                &config,
                &filter,
                |id| neighbors[id as usize].clone(),
                |id| distances[id as usize],
                0,
            )
            .unwrap();
            let _ = acorn_search_with_stats(
                3,
                &config,
                &filter,
                |id| neighbors[id as usize].clone(),
                |id| distances[id as usize],
                0,
            )
            .unwrap();
        }

        // Run interleaved (one of each per round) to even out any
        // remaining timing-source drift across the long measurement
        // window.
        let mut acc_no_stats_ns = 0u128;
        let mut acc_with_stats_ns = 0u128;
        let mut sink = 0u32;
        for _ in 0..iters {
            let t = Instant::now();
            let r = acorn_search(
                3,
                &config,
                &filter,
                |id| neighbors[id as usize].clone(),
                |id| distances[id as usize],
                0,
            )
            .unwrap();
            acc_no_stats_ns += t.elapsed().as_nanos();
            sink = sink.wrapping_add(r.len() as u32);

            let t = Instant::now();
            let (r, _stats) = acorn_search_with_stats(
                3,
                &config,
                &filter,
                |id| neighbors[id as usize].clone(),
                |id| distances[id as usize],
                0,
            )
            .unwrap();
            acc_with_stats_ns += t.elapsed().as_nanos();
            sink = sink.wrapping_add(r.len() as u32);
        }
        let no_stats_ns_per = acc_no_stats_ns as f64 / iters as f64;
        let with_stats_ns_per = acc_with_stats_ns as f64 / iters as f64;

        // Force the compiler to keep both loops live.
        std::hint::black_box(sink);

        println!("# acorn_stats_overhead_probe ({} iters)", iters);
        println!("# acorn_search             {:.0} ns/op", no_stats_ns_per);
        println!("# acorn_search_with_stats  {:.0} ns/op", with_stats_ns_per);
        println!(
            "# delta                     {:+.0} ns/op ({:+.1}%)",
            with_stats_ns_per - no_stats_ns_per,
            100.0 * (with_stats_ns_per - no_stats_ns_per) / no_stats_ns_per
        );
    }
}
