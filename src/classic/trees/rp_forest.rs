//! Random Projection Tree Forest implementation.
//!
//! Pure Rust implementation of a Random Projection Tree Forest.
//!
//! Note: some ecosystems refer to this family as “Annoy-style”, but this crate uses
//! method-family names (`rptree`) rather than vendor/library names.
//!
//! Algorithm:
//! - Forest of independent two-means trees
//! - Two-means hyperplane splits at each node, retried when too imbalanced
//! - Multiple trees improve recall through ensemble search
//! - Memory-mapped index support (optional)
//! - Thread-safe search
//!
//! **Relationships**:
//! - Uses Random Projection Trees (RP-Trees) as base structure
//! - Forest approach (multiple trees) improves recall vs single tree
//! - Complementary to graph-based methods (HNSW, SNG)
//! - Tree-based space partitioning (different from graph-based proximity)
//!
//! **vs hash-bucket LSH**: if you need near-duplicate detection or set similarity
//! (MinHash/SimHash) rather than ranked k-NN retrieval, consider a locality-sensitive
//! hashing library that provides banding LSH and random-projection LSH over hash buckets.
//!
//! # References
//!
//! - Dasgupta & Freund (2008): "Random projection trees and low dimensional manifolds"
//! - Spotify Engineering Blog: "Annoy: Approximate Nearest Neighbors in C++/Python" (historical name)

use crate::classic::trees::persistence::{read_json, validate_vector_shape, write_json_atomic};
use crate::simd;
use crate::RetrieveError;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;

const RP_FOREST_FORMAT_VERSION: u32 = 1;

/// Largest frontier reservation per query; see `search_with_budget_stats`.
const FRONTIER_RESERVE: usize = 4_096;
const DEFAULT_RP_FOREST_SEED: u64 = 0xA24B_AED4_963E_E407;

/// Candidate assignments the two-means fit makes per split attempt.
///
/// This is the reference implementation's value: the fit samples points with
/// replacement rather than sweeping the node, which keeps a split's cost
/// independent of how many points the node holds.
const TWO_MEANS_ITERATIONS: usize = 200;

/// Split attempts before the most balanced attempt is accepted.
const SPLIT_ATTEMPTS: usize = 3;

/// A split is accepted outright when its larger child holds less than this
/// fraction of the node's points.
const SPLIT_MAX_IMBALANCE: f32 = 0.95;

/// Above this imbalance no attempt is usable: the split leaves (at least) one
/// side effectively empty, so the node becomes a leaf instead of recursing.
const SPLIT_GIVE_UP_IMBALANCE: f32 = 0.99;

/// Random Projection Tree Forest index.
///
/// Uses a forest of independent random projection trees for approximate
/// nearest neighbor search. Each tree partitions space with hyperplanes fitted
/// by two-means to its own points.
///
#[derive(Deserialize, Serialize)]
pub struct RpForestIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    doc_ids: Vec<u32>,
    params: RpForestParams,
    built: bool,

    /// Forest of random projection trees
    pub(crate) trees: Vec<RPTree>,

    /// Largest squared L2 norm among the indexed vectors.
    ///
    /// Search derives its traversal bound from this (see
    /// [`RpForestIndex::search_with_budget`]), so it is recomputed from
    /// `vectors` on [`RpForestIndex::build`] and
    /// [`RpForestIndex::load_from_dir`] and never trusted from disk: a stale
    /// value would make the bound optimistic. The fallback is the family's
    /// documented contract (unit-normalised vectors); a `Deserialize` that
    /// bypasses `load_from_dir` inherits it.
    #[serde(skip, default = "unit_norm_bound")]
    pub(crate) max_sq_norm: f32,
}

/// Random Projection Tree Forest parameters.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RpForestParams {
    /// Number of trees in forest
    pub num_trees: usize,

    /// Tree construction parameters
    pub tree_params: RPTreeParams,
}

impl Default for RpForestParams {
    fn default() -> Self {
        Self {
            num_trees: 10,
            tree_params: RPTreeParams::default(),
        }
    }
}

/// Random projection tree.
#[derive(Clone, Deserialize, Serialize)]
pub(crate) struct RPTree {
    root: Option<TreeNode>,
}

/// Tree node.
#[derive(Clone, Deserialize, Serialize)]
enum TreeNode {
    Leaf {
        indices: Vec<u32>,
    },
    Internal {
        hyperplane: Vec<f32>, // Random hyperplane (unit normal)
        threshold: f32,       // Projection value the split sits on
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

/// Random projection tree parameters.
///
/// The derived `Default` leaves [`Self::max_leaf_size`] at `0`, which selects
/// the dimension-derived leaf size rather than a hardcoded cap.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct RPTreeParams {
    /// Maximum points per leaf, or `0` to derive it from the dimension with
    /// [`derived_leaf_size`] (the default).
    pub max_leaf_size: usize,
}

impl RPTreeParams {
    /// Leaf capacity to build with for `dimension`.
    ///
    /// A configured [`Self::max_leaf_size`] always wins; `0` is the sentinel for
    /// the dimension-derived default, which keeps existing
    /// `RPTreeParams { max_leaf_size: n }` callers working unchanged.
    #[must_use]
    pub fn leaf_size(&self, dimension: usize) -> usize {
        if self.max_leaf_size == 0 {
            derived_leaf_size(dimension)
        } else {
            self.max_leaf_size
        }
    }
}

/// Leaf capacity derived from the dimension, matching the sibling
/// implementation this family is measured against.
///
/// That layout stores one node per fixed-size block: a descendant count, two
/// child slots, and room for one `f32`-per-dimension vector. A leaf reuses the
/// block *after* the descendant count as a list of `u32` ids, so the number of
/// ids that fit is
///
/// ```text
/// (two child slots + dimension floats) / one id
/// = (2 * size_of::<u32>() + dimension * size_of::<f32>()) / size_of::<u32>()
/// = 2 + dimension
/// ```
///
/// because `f32` and `u32` are both four bytes. At `d = 96` that is 98 ids per
/// leaf, versus the flat 10 this family used to hardcode, so the leaf structure
/// — and the overshoot a budget pays when a leaf is opened — matches the
/// reference implementation's.
#[must_use]
pub fn derived_leaf_size(dimension: usize) -> usize {
    2 + dimension * std::mem::size_of::<f32>() / std::mem::size_of::<u32>()
}

#[derive(Deserialize, Serialize)]
struct RpForestSnapshot {
    version: u32,
    index: RpForestIndex,
}

impl RpForestIndex {
    /// Create a new RP-forest index.
    pub fn new(dimension: usize, params: RpForestParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            doc_ids: Vec::new(),
            params,
            built: false,
            trees: Vec::new(),
            max_sq_norm: unit_norm_bound(),
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "Cannot add vectors after index is built".to_string(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }

        self.vectors.extend_from_slice(&vector);
        self.doc_ids.push(doc_id);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index with random projection tree forest.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Build forest of trees
        self.trees = Vec::new();
        for tree_idx in 0..self.params.num_trees {
            let tree = self.build_tree(tree_seed(tree_idx))?;
            self.trees.push(tree);
        }

        self.refresh_norm_bound();
        self.built = true;
        Ok(())
    }

    /// Save a built RP-forest index to a directory.
    pub fn save_to_dir(&self, output_dir: impl AsRef<Path>) -> Result<(), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot save unbuilt RP-forest index".into(),
            ));
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;
        write_json_atomic(
            &output_dir.join("index.json"),
            &RpForestSnapshot {
                version: RP_FOREST_FORMAT_VERSION,
                index: self.clone_for_snapshot(),
            },
        )
    }

    /// Load an RP-forest index saved by [`Self::save_to_dir`].
    pub fn load_from_dir(input_dir: impl AsRef<Path>) -> Result<Self, RetrieveError> {
        let snapshot: RpForestSnapshot = read_json(&input_dir.as_ref().join("index.json"))?;
        if snapshot.version != RP_FOREST_FORMAT_VERSION {
            return Err(RetrieveError::FormatError(format!(
                "unsupported RP-forest format version {}",
                snapshot.version
            )));
        }
        let index = snapshot.index;
        validate_vector_shape(
            "RP-forest",
            index.dimension,
            index.num_vectors,
            &index.vectors,
            &index.doc_ids,
        )?;
        if !index.built || index.trees.is_empty() {
            return Err(RetrieveError::FormatError(
                "RP-forest snapshot is not built".into(),
            ));
        }
        let mut index = index;
        index.refresh_norm_bound();
        Ok(index)
    }

    fn clone_for_snapshot(&self) -> Self {
        Self {
            vectors: self.vectors.clone(),
            dimension: self.dimension,
            num_vectors: self.num_vectors,
            doc_ids: self.doc_ids.clone(),
            params: self.params.clone(),
            built: self.built,
            trees: self.trees.clone(),
            max_sq_norm: self.max_sq_norm,
        }
    }

    /// Estimated heap memory used by this index.
    pub fn memory_usage(&self) -> crate::memory::MemoryReport {
        let tree_storage = self.trees.capacity() * std::mem::size_of::<RPTree>();
        let node_storage: usize = self.trees.iter().map(RPTree::owned_bytes).sum();
        crate::memory::MemoryReport {
            vectors_bytes: self.vectors.capacity() * std::mem::size_of::<f32>(),
            graph_bytes: tree_storage + node_storage,
            quantized_bytes: 0,
            metadata_bytes: self.doc_ids.capacity() * std::mem::size_of::<u32>(),
        }
    }

    /// Build a single random projection tree.
    fn build_tree(&self, seed: u64) -> Result<RPTree, RetrieveError> {
        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        let root = self.build_tree_recursive(&indices, seed)?;
        Ok(RPTree { root })
    }

    /// Build tree recursively. Each call fits its own two-means split.
    fn build_tree_recursive(
        &self,
        indices: &[u32],
        seed: u64,
    ) -> Result<Option<TreeNode>, RetrieveError> {
        if indices.is_empty() {
            return Ok(None);
        }

        // Leaf node if small enough
        if indices.len() <= self.params.tree_params.leaf_size(self.dimension) {
            return Ok(Some(TreeNode::Leaf {
                indices: indices.to_vec(),
            }));
        }

        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // One seeded stream per node: the fit and any retries draw from it in a
        // fixed order, so a build is reproducible for a given seed.
        let mut rng = StdRng::seed_from_u64(seed);
        let Some((hyperplane, left_indices, right_indices)) =
            self.two_means_split(indices, &mut rng)
        else {
            // No usable hyperplane (e.g. duplicate points, where every centroid
            // pair collapses to one): keep the points together as a leaf rather
            // than recursing forever.
            return Ok(Some(TreeNode::Leaf {
                indices: indices.to_vec(),
            }));
        };

        // Each child independently picks its own split hyperplane
        let left = self.build_tree_recursive(&left_indices, child_seed(seed, 0))?;
        let right = self.build_tree_recursive(&right_indices, child_seed(seed, 1))?;

        Ok(Some(TreeNode::Internal {
            hyperplane,
            threshold: 0.0,
            left: Box::new(left.unwrap_or(TreeNode::Leaf {
                indices: Vec::new(),
            })),
            right: Box::new(right.unwrap_or(TreeNode::Leaf {
                indices: Vec::new(),
            })),
        }))
    }

    /// Fit a two-means split to `indices` and partition them by it.
    ///
    /// Attempts up to [`SPLIT_ATTEMPTS`] fits, taking the first whose larger
    /// child holds less than [`SPLIT_MAX_IMBALANCE`] of the node. Returns `None`
    /// when no attempt yields a usable hyperplane, which happens when the points
    /// are duplicates or collinear enough that both centroids collapse onto one
    /// another; the caller then keeps them together as a leaf.
    ///
    /// Unlike the reference implementation this never falls back to random
    /// sides: a random partition is not a hyperplane, so `far_child_dot_bound`
    /// would no longer bound it. The most balanced attempt is kept instead when
    /// it still separates the node, and only an effectively-empty side (above
    /// [`SPLIT_GIVE_UP_IMBALANCE`]) becomes a leaf.
    fn two_means_split(
        &self,
        indices: &[u32],
        rng: &mut rand::rngs::StdRng,
    ) -> Option<(Vec<f32>, Vec<u32>, Vec<u32>)> {
        let mut best: Option<(f32, Vec<f32>, Vec<u32>, Vec<u32>)> = None;
        for _ in 0..SPLIT_ATTEMPTS {
            // A collapsed fit (all points on one centroid) uses up an attempt but
            // must not discard a balanced one an earlier attempt already found.
            let Some(normal) = self.two_means_normal(indices, rng) else {
                continue;
            };
            let (left, right) = self.partition_on(indices, &normal);
            if left.is_empty() || right.is_empty() {
                continue;
            }
            let imbalance = split_imbalance(left.len(), right.len());
            if imbalance < SPLIT_MAX_IMBALANCE {
                return Some((normal, left, right));
            }
            if best.as_ref().is_none_or(|(b, ..)| imbalance < *b) {
                best = Some((imbalance, normal, left, right));
            }
        }
        match best {
            Some((imbalance, normal, left, right)) if imbalance <= SPLIT_GIVE_UP_IMBALANCE => {
                Some((normal, left, right))
            }
            _ => None,
        }
    }

    /// Route every id in `indices` by the sign of `dot(v, normal)`.
    ///
    /// Points exactly on the plane (`dot == 0`) go right, deterministically. Both
    /// branches of the tree are therefore non-empty only when the plane actually
    /// separates the node, which the caller checks.
    fn partition_on(&self, indices: &[u32], normal: &[f32]) -> (Vec<u32>, Vec<u32>) {
        let mut left = Vec::with_capacity(indices.len() / 2);
        let mut right = Vec::with_capacity(indices.len() / 2);
        for &idx in indices {
            if simd::dot(self.get_vector(idx as usize), normal) < 0.0 {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }
        (left, right)
    }

    /// Angular two-means: seed two centroids from distinct points, then move
    /// them toward [`TWO_MEANS_ITERATIONS`] randomly sampled points of the node,
    /// weighting each candidate's distance by how many points its centroid
    /// already owns so the two groups stay balanced.
    ///
    /// The returned normal is `p - q`, unit length. For the angular metric the
    /// decision boundary between two centroids passes through the origin:
    /// `dot(u, p) > dot(u, q)` iff `dot(u, p - q) > 0`. So the split is a
    /// hyperplane through the origin and the stored threshold stays `0.0` —
    /// the exact shape [`far_child_dot_bound`] is derived for. A non-zero
    /// threshold would also be sound (that bound takes one), but it would route
    /// by a rule the angular comparison does not justify.
    fn two_means_normal(&self, indices: &[u32], rng: &mut rand::rngs::StdRng) -> Option<Vec<f32>> {
        use rand::Rng as _;

        let count = indices.len();
        if count < 2 {
            return None;
        }
        let i = rng.random_range(0..count);
        let mut j = rng.random_range(0..count - 1);
        if j >= i {
            j += 1;
        }
        let mut p = unit_or_zero(self.get_vector(indices[i] as usize));
        let mut q = unit_or_zero(self.get_vector(indices[j] as usize));
        let (mut p_count, mut q_count) = (1usize, 1usize);

        for _ in 0..TWO_MEANS_ITERATIONS {
            let v = self.get_vector(indices[rng.random_range(0..count)] as usize);
            let norm = simd::dot(v, v).max(0.0).sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                continue;
            }
            // Weighting by the owned count is the reference implementation's
            // balancing heuristic: a centroid with more points resists taking
            // another one, which is what keeps the split from collapsing.
            let d_p = p_count as f32 * angular_distance(&p, v);
            let d_q = q_count as f32 * angular_distance(&q, v);
            if d_p < d_q {
                blend_toward(&mut p, v, norm, p_count);
                p_count += 1;
            } else if d_q < d_p {
                blend_toward(&mut q, v, norm, q_count);
                q_count += 1;
            }
        }

        let mut normal: Vec<f32> = p.iter().zip(&q).map(|(a, b)| a - b).collect();
        let norm = simd::dot(&normal, &normal).max(0.0).sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return None;
        }
        for x in &mut normal {
            *x /= norm;
        }
        Some(normal)
    }

    /// Search for k nearest neighbors.
    ///
    /// Uses Annoy's default candidate budget, `k * num_trees`. Call
    /// [`Self::search_with_budget`] to trade recall for latency explicitly.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_budget(query, k, k.saturating_mul(self.params.num_trees))
    }

    /// Search for k nearest neighbors with an explicit candidate budget.
    ///
    /// The traversal is Annoy's: one global max-heap over
    /// `(bound, node)` entries, every tree root pushed with `+inf` so the
    /// forest interleaves instead of descending tree by tree, and the loop
    /// stopping once `search_k` distinct candidate ids are held. A leaf
    /// contributes all of its ids at once, so the collected count overshoots
    /// `search_k` by at most `max_leaf_size - 1`.
    ///
    /// Both children of every split are pushed unconditionally. The child the
    /// query falls into keeps the parent's bound; the other one is pushed with
    /// a strictly smaller bound, which makes near-before-far a consequence of
    /// heap order rather than an `if`. See `far_child_dot_bound` for the
    /// derivation, and `TraversalEntry` for how the margin Annoy pushes is
    /// retired to a tie-break so that the ordering stays conservative.
    pub fn search_with_budget(
        &self,
        query: &[f32],
        k: usize,
        search_k: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        self.search_with_budget_stats(query, k, search_k)
            .map(|(results, _)| results)
    }

    /// [`Self::search_with_budget`] plus the traversal counters.
    ///
    /// The counters exist because the budget is spent per leaf, not per id:
    /// `candidates` is how many ids were actually scored, which is what
    /// `search_k` is really parameterising.
    pub fn search_with_budget_stats(
        &self,
        query: &[f32],
        k: usize,
        search_k: usize,
    ) -> Result<(Vec<(u32, f32)>, RpForestSearchStats), RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "Index must be built before search".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let mut stats = RpForestSearchStats::default();
        if k == 0 || search_k == 0 {
            return Ok((Vec::new(), stats));
        }

        // Largest dot product any indexed vector can reach against this query.
        // Everything below tightens this; it is the bound inherited by the
        // child the query falls into.
        let query_norm_sq = simd::dot(query, query);
        let query_norm = query_norm_sq.max(0.0).sqrt();
        let radius_sq = self.max_sq_norm.max(0.0);
        let radius = radius_sq.sqrt();
        let max_dot = radius * query_norm;

        // One internal node in, one net entry out, so the frontier ends up about
        // as wide as the number of splits visited; reserving that keeps `push`
        // from reallocating a dozen times per query. Capped, because a budget of
        // `n` must not reserve a frontier the size of the index.
        let mut heap: BinaryHeap<TraversalEntry<'_>> = BinaryHeap::with_capacity(
            search_k
                .min(FRONTIER_RESERVE)
                .saturating_add(self.trees.len()),
        );
        for tree in &self.trees {
            if let Some(root) = tree.root.as_ref() {
                heap.push(TraversalEntry {
                    bound: f32::INFINITY,
                    margin: f32::INFINITY,
                    node: root,
                });
            }
        }

        let leaf_size = self.params.tree_params.leaf_size(self.dimension);
        let room = search_k.saturating_add(leaf_size);
        let mut candidates: Vec<u32> = Vec::with_capacity(room);
        let mut seen: HashSet<u32> = HashSet::with_capacity(room);

        while candidates.len() < search_k {
            let Some(entry) = heap.pop() else { break };
            stats.nodes_expanded += 1;
            match entry.node {
                TreeNode::Leaf { indices } => {
                    stats.leaves_visited += 1;
                    for &idx in indices {
                        if seen.insert(idx) {
                            candidates.push(idx);
                        }
                    }
                }
                TreeNode::Internal {
                    hyperplane,
                    threshold,
                    left,
                    right,
                } => {
                    let projection = simd::dot(query, hyperplane);
                    // Same routing rule as the build.
                    let near_is_left = projection < *threshold;
                    let bound = entry.bound.min(max_dot);
                    let near_margin = entry.margin.min(projection.abs());
                    let (near, far, far_is_upper) = if near_is_left {
                        (left.as_ref(), right.as_ref(), true)
                    } else {
                        (right.as_ref(), left.as_ref(), false)
                    };
                    let far_bound = far_child_dot_bound(
                        projection,
                        *threshold,
                        radius,
                        radius_sq,
                        query_norm,
                        query_norm_sq,
                        far_is_upper,
                    )
                    .min(bound);
                    heap.push(TraversalEntry {
                        bound,
                        margin: near_margin,
                        node: near,
                    });
                    heap.push(TraversalEntry {
                        bound: far_bound,
                        // Annoy's key for the far child, sign flipped: it settles
                        // the ties the bound cannot separate, e.g. a query
                        // sitting on the split plane.
                        margin: entry.margin.min(-projection.abs()),
                        node: far,
                    });
                }
            }
        }
        stats.candidates = candidates.len();

        // Compute exact distances for candidates. Annoy sorts ids and skips
        // repeats here; the set above already removed them, and the ordering
        // that matters is the final sort, so both reduce to the same thing.
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .map(|idx| {
                let vec = self.get_vector(idx as usize);
                let dist = 1.0 - simd::dot(query, vec);
                (self.doc_ids[idx as usize], dist)
            })
            .collect();

        // Sort and return top k
        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        Ok((results.into_iter().take(k).collect(), stats))
    }

    /// Recompute the largest squared norm in the index.
    fn refresh_norm_bound(&mut self) {
        let mut max_sq = 0.0f32;
        for vector in self.vectors.chunks_exact(self.dimension) {
            let sq = simd::dot(vector, vector);
            if sq.is_finite() && sq > max_sq {
                max_sq = sq;
            }
        }
        self.max_sq_norm = max_sq;
    }

    /// Get vector from SoA storage.
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

/// Traversal counters for one [`RpForestIndex::search_with_budget_stats`] call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RpForestSearchStats {
    /// Distinct candidate ids collected and scored.
    pub candidates: usize,
    /// Leaf nodes popped, each contributing all of its ids.
    pub leaves_visited: usize,
    /// Heap pops, leaves and internal nodes together.
    pub nodes_expanded: usize,
}

/// One entry of the global traversal heap.
///
/// `bound` is a conservative upper bound on `query · v` over every vector in
/// the subtree, so the heap is an A\* frontier: it only expands a subtree with
/// a lower bound after every subtree that can do at least as well.
///
/// `margin` is Annoy's key: `min` over the path of `query · hyperplane`, negated
/// for the far child. On this family it is *not* an upper bound (the cap a split
/// carves out always contains directions arbitrarily close to the query), so it
/// cannot order the traversal on its own without deferring subtrees that hold
/// true neighbours. It is kept as a tie-break: among subtrees whose bounds are
/// equal it reproduces Annoy's ordering, and that is exactly the case that the
/// bound cannot separate, because the query lies inside the child it descends
/// into.
struct TraversalEntry<'a> {
    bound: f32,
    margin: f32,
    node: &'a TreeNode,
}

impl PartialEq for TraversalEntry<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for TraversalEntry<'_> {}

impl PartialOrd for TraversalEntry<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TraversalEntry<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound
            .total_cmp(&other.bound)
            .then_with(|| self.margin.total_cmp(&other.margin))
    }
}

/// Conservative upper bound on `query · u` for the half of a split the query
/// does not fall into.
///
/// Derivation, with `h` the split's unit normal, `t` its threshold, `R` the
/// largest vector norm in the index, `Q = |query|`, `beta = query · h`, and
/// `alpha = u · h`:
///
/// * Write `u = alpha h + u_perp` and `query = beta h + q_perp`. Then
///   `query · u = alpha beta + q_perp · u_perp`, and by Cauchy-Schwarz
///   `q_perp · u_perp <= |u_perp| |q_perp|`, with `|u_perp| <= sqrt(R^2 -
///   alpha^2)` and `|q_perp| = sqrt(Q^2 - beta^2)`. So
///   `query · u <= g(alpha) = alpha beta + sqrt((R^2 - alpha^2)(Q^2 - beta^2))`.
/// * The far half is `alpha >= t` or `alpha <= t`, a half-line in `alpha`
///   because the split is a hyperplane. `g` is unimodal on `[-R, R]` with its
///   peak at `alpha* = beta R / Q` (the direction of the query), so its
///   supremum over the far half is `g(alpha* clamped onto that half)`.
///
/// Every branch of the tree therefore has a bound that is at least the true
/// best dot inside it, and taking `min` with the parent's bound keeps the
/// sequence non-increasing down a path without losing that property (the
/// subtree's supremum cannot exceed its parent's). At any budget, nothing that
/// the traversal defers can hold a vector that it should have scored first.
///
/// For unit-normalised data (`R = Q = 1`) this reduces to
/// `t beta + sqrt((1 - t^2)(1 - beta^2))`. With `t = 0`, which is what this
/// build stores, that is the sine of the angle between the query and the split
/// normal, and it goes negative once the query and the far side are more than a
/// right angle apart.
///
/// The build stores `t = 0` by construction, not by approximation: the split
/// normal is the difference of two angular centroids, and the angular decision
/// boundary between centroids `p` and `q` is `dot(u, p) = dot(u, q)`, i.e. the
/// plane through the origin orthogonal to `p - q`. The offset the general
/// derivation above carries is therefore exactly zero. If a future split rule
/// stores a non-zero offset, this function already takes it as `threshold`; no
/// change is needed here, but the build must keep routing with the same
/// `projection < threshold` rule that search uses.
fn far_child_dot_bound(
    beta: f32,
    threshold: f32,
    radius: f32,
    radius_sq: f32,
    query_norm: f32,
    query_norm_sq: f32,
    far_is_upper_side: bool,
) -> f32 {
    if radius <= 0.0 || query_norm <= 0.0 {
        // Zero-length vectors (or query): every dot is zero, and the ordering
        // falls back to the margin tie-break.
        return 0.0;
    }
    if !radius.is_finite() || !query_norm.is_finite() {
        // Non-finite inputs (a NaN also lands here, since it compares false to
        // zero): give up on pruning rather than emit an unsound bound.
        return f32::INFINITY;
    }
    let threshold = threshold.clamp(-radius, radius);
    let peak = (beta * radius / query_norm).clamp(-radius, radius);
    let alpha = if far_is_upper_side {
        peak.max(threshold)
    } else {
        peak.min(threshold)
    };
    let perp_vectors = (radius_sq - alpha * alpha).max(0.0);
    let perp_query = (query_norm_sq - beta * beta).max(0.0);
    alpha * beta + (perp_vectors * perp_query).sqrt()
}

/// Norm bound assumed when the index has not been built or loaded yet.
fn unit_norm_bound() -> f32 {
    1.0
}

impl RPTree {
    fn owned_bytes(&self) -> usize {
        self.root.as_ref().map(TreeNode::owned_bytes).unwrap_or(0)
    }
}

impl TreeNode {
    fn owned_bytes(&self) -> usize {
        match self {
            TreeNode::Leaf { indices } => indices.capacity() * std::mem::size_of::<u32>(),
            TreeNode::Internal {
                hyperplane,
                left,
                right,
                ..
            } => {
                hyperplane.capacity() * std::mem::size_of::<f32>()
                    + boxed_node_bytes(left)
                    + boxed_node_bytes(right)
            }
        }
    }
}

fn boxed_node_bytes(node: &TreeNode) -> usize {
    std::mem::size_of::<TreeNode>() + node.owned_bytes()
}

/// Larger child's share of a split, in `[0.5, 1]`.
///
/// The epsilon is the reference implementation's guard against `0/0` on an
/// empty node.
fn split_imbalance(left: usize, right: usize) -> f32 {
    let total = (left + right) as f32;
    let share = left as f32 / (total + 1e-9);
    share.max(1.0 - share)
}

/// `2 - 2 cos(x, y)`, the angular distance used to assign points to centroids.
///
/// Zero-length operands are a cosine of zero by construction, matching the
/// reference implementation rather than producing a NaN.
fn angular_distance(x: &[f32], y: &[f32]) -> f32 {
    let xx = simd::dot(x, x);
    let yy = simd::dot(y, y);
    let xy = simd::dot(x, y);
    let ppqq = xx * yy;
    if ppqq > 0.0 {
        2.0 - 2.0 * xy / ppqq.sqrt()
    } else {
        2.0
    }
}

/// `v / |v|`, or `v` unchanged when it is zero.
fn unit_or_zero(v: &[f32]) -> Vec<f32> {
    let norm = simd::dot(v, v).max(0.0).sqrt();
    if norm > 0.0 && norm.is_finite() {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Move `centroid` one step toward the unit vector along `v`, keeping it the
/// mean of the `owned` points it already represents.
fn blend_toward(centroid: &mut [f32], v: &[f32], v_norm: f32, owned: usize) {
    let owned = owned as f32;
    for (c, x) in centroid.iter_mut().zip(v) {
        *c = (*c * owned + x / v_norm) / (owned + 1.0);
    }
}

fn tree_seed(tree_idx: usize) -> u64 {
    DEFAULT_RP_FOREST_SEED ^ (tree_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn child_seed(seed: u64, branch: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407 ^ branch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index(n: usize, dim: usize) -> RpForestIndex {
        let params = RpForestParams {
            num_trees: 5,
            tree_params: RPTreeParams { max_leaf_size: 10 },
        };
        let mut index = RpForestIndex::new(dim, params).unwrap();
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            v[i % dim] = 1.0;
            index.add(5000 + i as u32, v).unwrap();
        }
        index.build().unwrap();
        index
    }

    #[test]
    fn build_is_deterministic() {
        let first = build_index(50, 4);
        let second = build_index(50, 4);
        assert_eq!(
            serde_json::to_value(&first.trees).unwrap(),
            serde_json::to_value(&second.trees).unwrap()
        );
    }

    #[test]
    fn test_basic_search_returns_results() {
        let index = build_index(50, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_returns_at_most_k() {
        let index = build_index(50, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        for k in [1, 3, 5, 10] {
            let results = index.search(&query, k).unwrap();
            assert!(results.len() <= k);
        }
    }

    #[test]
    fn test_results_sorted_by_distance() {
        let index = build_index(50, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", results);
        }
    }

    #[test]
    fn test_ids_in_bounds() {
        let n = 50usize;
        let index = build_index(n, 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        for (id, _) in results {
            assert!(
                (5000..5000 + n as u32).contains(&id),
                "id {} out of shifted doc-id bounds (n={})",
                id,
                n
            );
        }
    }

    #[test]
    fn test_multiple_trees_improve_coverage() {
        // With many trees, the candidate set should cover a reasonable fraction of
        // close vectors. Regression guard: ensures each tree uses an independent
        // hyperplane (the bug was both children shared one hyperplane per level).
        let dim = 8;
        let n = 100;
        let params = RpForestParams {
            num_trees: 20,
            tree_params: RPTreeParams { max_leaf_size: 5 },
        };
        let mut index = RpForestIndex::new(dim, params).unwrap();
        // Add clustered data: half near [1,0,...], half near [0,1,0,...]
        for i in 0..n {
            let mut v = vec![0.0f32; dim];
            if i < n / 2 {
                v[0] = 1.0;
                v[1] = 0.01 * (i as f32);
            } else {
                v[1] = 1.0;
                v[0] = 0.01 * (i as f32);
            }
            // normalize
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }
            index.add(6000 + i as u32, v).unwrap();
        }
        index.build().unwrap();

        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = index.search(&query, 10).unwrap();
        // All top-10 should come from the [1,0,...] cluster (ids 0..50)
        let correct = results
            .iter()
            .filter(|(id, _)| (6000..6000 + (n / 2) as u32).contains(id))
            .count();
        assert!(
            correct >= 5,
            "only {correct}/10 results from correct cluster — hyperplane independence may be broken"
        );
    }

    #[test]
    fn test_build_errors_on_empty_index() {
        let mut index = RpForestIndex::new(4, RpForestParams::default()).unwrap();
        assert!(index.build().is_err());
    }

    #[test]
    fn test_add_after_build_errors() {
        let mut index = RpForestIndex::new(4, RpForestParams::default()).unwrap();
        index.add(0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        index.build().unwrap();
        assert!(index.add(1, vec![0.0, 1.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut index = RpForestIndex::new(4, RpForestParams::default()).unwrap();
        assert!(index.add(0, vec![1.0, 0.0]).is_err());
    }

    #[test]
    fn save_load_roundtrip_preserves_sampled_forest() {
        let index = build_index(50, 4);
        let dir = tempfile::tempdir().unwrap();
        index.save_to_dir(dir.path()).unwrap();
        let loaded = RpForestIndex::load_from_dir(dir.path()).unwrap();
        let query = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(
            index.search(&query, 8).unwrap(),
            loaded.search(&query, 8).unwrap()
        );
    }

    #[test]
    fn test_degenerate_split_does_not_recurse_infinitely() {
        // All vectors identical → both centroids are that same point, so the
        // fit collapses and no split normal exists.
        // Regression guard: the degenerate-split guard must fall back to a leaf node
        // rather than recursing until stack overflow.
        let params = RpForestParams {
            num_trees: 3,
            tree_params: RPTreeParams { max_leaf_size: 2 },
        };
        let mut index = RpForestIndex::new(4, params).unwrap();
        for i in 0..20u32 {
            index.add(i, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        }
        // Must complete without stack overflow.
        index.build().unwrap();
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.is_empty());
    }

    // -- budgeted traversal ---------------------------------------------------

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn seeded_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    fn forest(vectors: &[Vec<f32>], dim: usize, num_trees: usize, leaf: usize) -> RpForestIndex {
        let params = RpForestParams {
            num_trees,
            tree_params: RPTreeParams {
                max_leaf_size: leaf,
            },
        };
        let mut index = RpForestIndex::new(dim, params).unwrap();
        for (i, v) in vectors.iter().enumerate() {
            index.add(i as u32, v.clone()).unwrap();
        }
        index.build().unwrap();
        index
    }

    fn brute_force(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut scored: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, 1.0 - simd::dot(query, v)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    fn recall_of(vectors: &[Vec<f32>], got: &[(u32, f32)], query: &[f32], k: usize) -> f32 {
        let gt = brute_force(vectors, query, k);
        let hits = got
            .iter()
            .filter(|(id, _)| gt.iter().any(|(g, _)| g == id))
            .count();
        hits as f32 / gt.len() as f32
    }

    #[test]
    fn full_budget_matches_brute_force() {
        // A budget that can reach every leaf must reproduce exhaustive search
        // exactly, including the (distance, doc_id) tie order.
        let n = 2_000;
        let dim = 32;
        let k = 10;
        let vectors = seeded_unit_vectors(n, dim, 11);
        let index = forest(&vectors, dim, 6, 12);
        let queries = seeded_unit_vectors(8, dim, 22);
        for (qi, query) in queries.iter().enumerate() {
            let approx = index.search_with_budget(query, k, n).unwrap();
            let exact = brute_force(&vectors, query, k);
            assert_eq!(approx, exact, "query {qi} diverged from brute force");
        }
    }

    #[test]
    fn recall_is_monotone_in_the_budget() {
        let n = 2_000;
        let dim = 24;
        let k = 10;
        let vectors = seeded_unit_vectors(n, dim, 3);
        let index = forest(&vectors, dim, 10, 10);
        let queries = seeded_unit_vectors(20, dim, 4);
        let mut previous = -1.0f32;
        for budget in [50usize, 100, 200, 400, 800, 1_600, n] {
            let mut recall = 0.0f32;
            for query in &queries {
                let got = index.search_with_budget(query, k, budget).unwrap();
                recall += recall_of(&vectors, &got, query, k);
            }
            recall /= queries.len() as f32;
            assert!(
                recall >= previous - 1e-6,
                "recall fell from {previous:.3} to {recall:.3} at search_k={budget}"
            );
            previous = recall;
        }
        assert!(
            (previous - 1.0).abs() < 1e-6,
            "full budget recall {previous:.3} != 1.0"
        );
    }

    #[test]
    fn budget_overshoot_stays_within_one_leaf() {
        // Annoy checks the id count between pops, so the last leaf can push the
        // count past search_k: by at most its own size, never more.
        let n = 2_000;
        let dim = 24;
        let k = 10;
        let leaf = 16;
        let vectors = seeded_unit_vectors(n, dim, 5);
        let index = forest(&vectors, dim, 6, leaf);
        for budget in [10usize, 50, 99, 100, 333, 700] {
            for query in seeded_unit_vectors(4, dim, 6).iter() {
                let (_, stats) = index.search_with_budget_stats(query, k, budget).unwrap();
                assert!(
                    stats.candidates >= budget,
                    "search_k={budget} collected only {} ids",
                    stats.candidates
                );
                assert!(
                    stats.candidates < budget + leaf,
                    "search_k={budget} collected {} ids, more than one leaf of overshoot",
                    stats.candidates
                );
            }
        }
    }

    #[test]
    fn far_child_bound_covers_its_half_space() {
        // Sample the attaining point as well as random ones: the bound must sit
        // on top of the cap's supremum, not merely above an arbitrary sample.
        let dim = 8;
        let mut rng = StdRng::seed_from_u64(99);
        for _ in 0..100 {
            let mut h: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let norm = h.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut h {
                *x /= norm;
            }
            // Orthonormal helper directions e1, e2 spanning the plane around h.
            let (e1, e2) = orthonormal_pair(&h);
            let beta = rng.random::<f32>() * 1.6 - 0.8;
            let q_perp = (1.0 - beta * beta).max(0.0).sqrt();
            let phi_q = rng.random::<f32>() * std::f32::consts::TAU;
            let query: Vec<f32> = (0..dim)
                .map(|d| beta * h[d] + q_perp * (phi_q.cos() * e1[d] + phi_q.sin() * e2[d]))
                .collect();
            let query_norm_sq = simd::dot(&query, &query);
            for &threshold in &[-0.6f32, -0.2, 0.0, 0.3, 0.75] {
                for &far_is_upper in &[true, false] {
                    let bound = far_child_dot_bound(
                        beta,
                        threshold,
                        1.0,
                        1.0,
                        query_norm_sq.sqrt(),
                        query_norm_sq,
                        far_is_upper,
                    );
                    // Points in the far cap, phase-aligned with the query (the
                    // worst case) and randomly rotated.
                    for &alpha in &[-0.99f32, -0.5, 0.0, 0.4, 0.9] {
                        if (far_is_upper && alpha < threshold)
                            || (!far_is_upper && alpha > threshold)
                        {
                            continue;
                        }
                        let u_perp = (1.0 - alpha * alpha).max(0.0).sqrt();
                        for &phi in &[
                            phi_q,
                            phi_q + 0.7,
                            rng.random::<f32>() * std::f32::consts::TAU,
                        ] {
                            let u: Vec<f32> = (0..dim)
                                .map(|d| {
                                    alpha * h[d] + u_perp * (phi.cos() * e1[d] + phi.sin() * e2[d])
                                })
                                .collect();
                            let dot = simd::dot(&query, &u);
                            assert!(
                                dot <= bound + 1e-4,
                                "dot {dot} exceeded bound {bound} (alpha={alpha}, t={threshold})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn far_child_bound_holds_for_non_unit_norms() {
        // Same bound with unnormalised data: the index radius and the query
        // norm are what keep it an upper bound once |u| != 1.
        let dim = 6;
        let radius = 1.7f32;
        let mut rng = StdRng::seed_from_u64(1234);
        for _ in 0..100 {
            let mut h: Vec<f32> = (0..dim).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let norm = h.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut h {
                *x /= norm;
            }
            let (e1, e2) = orthonormal_pair(&h);
            let query_scale = rng.random::<f32>() * 3.0 + 0.1;
            let b = rng.random::<f32>() * 1.8 - 0.9;
            let q_perp = (1.0 - b * b).max(0.0).sqrt();
            let phi_q = rng.random::<f32>() * std::f32::consts::TAU;
            let query: Vec<f32> = (0..dim)
                .map(|d| {
                    query_scale * (b * h[d] + q_perp * (phi_q.cos() * e1[d] + phi_q.sin() * e2[d]))
                })
                .collect();
            let query_norm_sq = simd::dot(&query, &query);
            let query_norm = query_norm_sq.sqrt();
            let beta = simd::dot(&query, &h);
            for &threshold in &[-1.2f32, -0.4, 0.0, 0.7, 1.5] {
                for &far_is_upper in &[true, false] {
                    let bound = far_child_dot_bound(
                        beta,
                        threshold,
                        radius,
                        radius * radius,
                        query_norm,
                        query_norm_sq,
                        far_is_upper,
                    );
                    for &scale in &[0.2f32, 0.9, radius] {
                        for &a in &[-0.95f32, -0.3, 0.0, 0.5, 0.95] {
                            let projection = scale * a;
                            if (far_is_upper && projection < threshold)
                                || (!far_is_upper && projection > threshold)
                            {
                                continue;
                            }
                            let u_perp = (1.0 - a * a).max(0.0).sqrt() * scale;
                            for &phi in &[phi_q, phi_q + 1.1] {
                                let u: Vec<f32> = (0..dim)
                                    .map(|d| {
                                        scale * a * h[d]
                                            + u_perp * (phi.cos() * e1[d] + phi.sin() * e2[d])
                                    })
                                    .collect();
                                let dot = simd::dot(&query, &u);
                                assert!(
                                    dot <= bound + 1e-3,
                                    "dot {dot} exceeded bound {bound} (|u|={scale}, t={threshold})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Orthonormal `(e1, e2)` completing `h` to a basis, so tests can place
    /// points at a chosen projection `alpha` and phase `phi` around `h`.
    fn orthonormal_pair(h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut axes: Vec<usize> = (0..h.len()).collect();
        axes.sort_by(|&a, &b| h[a].abs().total_cmp(&h[b].abs()));
        let e1 = orthogonalize(axis(h.len(), axes[0]), &[h]);
        let e2 = orthogonalize(axis(h.len(), axes[1]), &[h, &e1]);
        (e1, e2)
    }

    fn axis(dim: usize, at: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[at] = 1.0;
        v
    }

    fn orthogonalize(mut v: Vec<f32>, basis: &[&[f32]]) -> Vec<f32> {
        for b in basis {
            let projection: f32 = v.iter().zip(*b).map(|(a, c)| a * c).sum();
            for (x, c) in v.iter_mut().zip(*b) {
                *x -= projection * c;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    /// Random vectors with norms spread over `[0.2, 2.0]`, so the bound is
    /// exercised with `R != 1` rather than only on the unit-normalised contract.
    fn seeded_scaled_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let scale = 0.2 + 1.8 * rng.random::<f32>();
                (0..dim)
                    .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                    .collect()
            })
            .collect()
    }

    /// Every id in `node`, for checking a bound against the true maximum dot.
    fn subtree_ids(node: &TreeNode, out: &mut Vec<u32>) {
        match node {
            TreeNode::Leaf { indices } => out.extend_from_slice(indices),
            TreeNode::Internal { left, right, .. } => {
                subtree_ids(left, out);
                subtree_ids(right, out);
            }
        }
    }

    fn subtree_max_dot(node: &TreeNode, vectors: &[Vec<f32>], query: &[f32]) -> f32 {
        let mut ids = Vec::new();
        subtree_ids(node, &mut ids);
        ids.iter()
            .map(|&i| simd::dot(query, &vectors[i as usize]))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Walk every split and assert the bound the traversal would assign to the
    /// far child is at least the largest dot actually inside it.
    ///
    /// This is the property the priority queue relies on: a subtree parked with
    /// `bound` cannot contain a vector closer than one expanded with a larger
    /// bound. A two-means normal is still a hyperplane through the origin, so
    /// this must hold with `threshold = 0` and with non-unit vectors.
    fn assert_bounds_cover(
        node: &TreeNode,
        vectors: &[Vec<f32>],
        query: &[f32],
        radius: f32,
        radius_sq: f32,
        query_norm: f32,
        query_norm_sq: f32,
        inherited: f32,
    ) {
        let TreeNode::Internal {
            hyperplane,
            threshold,
            left,
            right,
        } = node
        else {
            return;
        };
        let projection = simd::dot(query, hyperplane);
        let near_is_left = projection < *threshold;
        let (near, far, far_is_upper) = if near_is_left {
            (left.as_ref(), right.as_ref(), true)
        } else {
            (right.as_ref(), left.as_ref(), false)
        };
        let far_bound = far_child_dot_bound(
            projection,
            *threshold,
            radius,
            radius_sq,
            query_norm,
            query_norm_sq,
            far_is_upper,
        )
        .min(inherited);
        let far_max = subtree_max_dot(far, vectors, query);
        assert!(
            far_bound.is_finite(),
            "bound must stay finite for finite input (projection={projection}, t={threshold})"
        );
        assert!(
            far_max <= far_bound + 1e-4,
            "far child holds dot {far_max} but was parked at bound {far_bound} \
             (projection={projection}, t={threshold}, far_is_upper={far_is_upper})"
        );
        assert!(
            subtree_max_dot(near, vectors, query) <= inherited + 1e-4,
            "near child exceeded the inherited bound {inherited}"
        );
        assert_bounds_cover(
            near,
            vectors,
            query,
            radius,
            radius_sq,
            query_norm,
            query_norm_sq,
            inherited,
        );
        assert_bounds_cover(
            far,
            vectors,
            query,
            radius,
            radius_sq,
            query_norm,
            query_norm_sq,
            far_bound,
        );
    }

    #[test]
    fn two_means_bound_covers_every_far_child() {
        for dim in [3usize, 8, 32] {
            for seed in [1u64, 7] {
                let vectors = seeded_scaled_vectors(700, dim, seed);
                let index = forest(&vectors, dim, 3, 0);
                for query in seeded_scaled_vectors(6, dim, seed + 100).iter() {
                    let query_norm_sq = simd::dot(query, query);
                    let query_norm = query_norm_sq.max(0.0).sqrt();
                    let radius_sq = index.max_sq_norm.max(0.0);
                    let radius = radius_sq.sqrt();
                    // The traversal starts every root at `+inf` and immediately
                    // tightens to `R * Q`; start there so the root's near-child
                    // check is the same Cauchy-Schwarz bound search uses.
                    let root_bound = radius * query_norm;
                    for tree in &index.trees {
                        if let Some(root) = tree.root.as_ref() {
                            assert_bounds_cover(
                                root,
                                &vectors,
                                query,
                                radius,
                                radius_sq,
                                query_norm,
                                query_norm_sq,
                                root_bound,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn split_partition_covers_every_point_once() {
        // A split that drops or duplicates an id would hide a vector from every
        // query regardless of budget. Check the partition directly, including
        // on the duplicate-heavy data that forces the leaf fallback.
        let dim = 12;
        let n = 900;
        let mut vectors = seeded_unit_vectors(n, dim, 41);
        // Repeat a block of points so degenerate splits are in play.
        for i in 0..120 {
            vectors[i + 300] = vectors[i].clone();
            vectors[i + 600] = vectors[i].clone();
        }
        let index = forest(&vectors, dim, 4, 0);
        for tree in &index.trees {
            let mut ids = Vec::new();
            subtree_ids(tree.root.as_ref().unwrap(), &mut ids);
            ids.sort_unstable();
            assert_eq!(
                ids,
                (0..n as u32).collect::<Vec<_>>(),
                "tree is not a partition of the index"
            );
        }
    }

    /// Assert every stored split satisfies the bound's preconditions: a unit
    /// normal and a plane through the origin.
    fn assert_split_preconditions(node: &TreeNode) {
        if let TreeNode::Internal {
            hyperplane,
            threshold,
            left,
            right,
        } = node
        {
            let norm = simd::dot(hyperplane, hyperplane).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "split normal is not unit length ({norm})"
            );
            assert_eq!(*threshold, 0.0, "two-means splits are through the origin");
            assert_split_preconditions(left);
            assert_split_preconditions(right);
        }
    }

    #[test]
    fn split_guard_never_returns_a_degenerate_partition() {
        // 999 duplicates plus one far outlier: a two-means fit either collapses
        // (both centroids on the duplicate) or isolates the outlier, which the
        // 0.99 give-up threshold rejects. Either way the node must stay a leaf
        // rather than recurse, and every id must remain reachable.
        let dim = 8;
        let mut vectors = vec![vec![0.0f32; dim]; 999];
        for v in &mut vectors {
            v[0] = 1.0;
        }
        let mut outlier = vec![0.0f32; dim];
        outlier[1] = 1.0;
        vectors.push(outlier);

        let index = forest(&vectors, dim, 2, 0);
        for tree in &index.trees {
            let mut ids = Vec::new();
            subtree_ids(tree.root.as_ref().unwrap(), &mut ids);
            ids.sort_unstable();
            assert_eq!(
                ids,
                (0..vectors.len() as u32).collect::<Vec<_>>(),
                "points lost on the degenerate split"
            );
            assert_split_preconditions(tree.root.as_ref().unwrap());
        }
        let got = index.search(&vectors[999], 1).unwrap();
        assert_eq!(got[0].0, 999, "the outlier must still be retrievable");
    }

    /// Walk a subtree, returning `(descendant count, leaf count, depth)`.
    fn subtree_shape(node: &TreeNode) -> (usize, usize, usize) {
        match node {
            TreeNode::Leaf { indices } => (indices.len(), 1, 0),
            TreeNode::Internal { left, right, .. } => {
                let (ln, ll, ld) = subtree_shape(left);
                let (rn, rl, rd) = subtree_shape(right);
                (ln + rn, ll + rl, 1 + ld.max(rd))
            }
        }
    }

    fn collect_split_stats(
        node: &TreeNode,
        imbalances: &mut Vec<f32>,
        leaf_sizes: &mut Vec<usize>,
        depths: &mut Vec<(usize, usize)>,
        depth: usize,
    ) {
        match node {
            TreeNode::Leaf { indices } => {
                leaf_sizes.push(indices.len());
                depths.push((depth, indices.len()));
            }
            TreeNode::Internal { left, right, .. } => {
                let (ln, ..) = subtree_shape(left);
                let (rn, ..) = subtree_shape(right);
                let total = (ln + rn) as f32;
                let f = ln as f32 / (total + 1e-9);
                imbalances.push(f.max(1.0 - f));
                collect_split_stats(left, imbalances, leaf_sizes, depths, depth + 1);
                collect_split_stats(right, imbalances, leaf_sizes, depths, depth + 1);
            }
        }
    }

    /// Every internal node's point set, for re-fitting and counting attempts.
    fn internal_node_sets(node: &TreeNode, out: &mut Vec<Vec<u32>>) {
        if let TreeNode::Internal { left, right, .. } = node {
            let mut ids = Vec::new();
            subtree_ids(left, &mut ids);
            subtree_ids(right, &mut ids);
            out.push(ids);
            internal_node_sets(left, out);
            internal_node_sets(right, out);
        }
    }

    fn pct(sorted: &[f32], p: f32) -> f32 {
        if sorted.is_empty() {
            return 0.0;
        }
        let i = ((sorted.len() - 1) as f32 * p).round() as usize;
        sorted[i]
    }

    /// Measurement, not a gate: the imbalance and occupancy of the splits the
    /// current rule produces on the two distributions the parity harness uses.
    ///
    /// The leaf cap is pinned to 98 (= `derived_leaf_size(96)`) so that runs
    /// before and after a split-rule change describe the same node set.
    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn report_split_balance() {
        let dim = 96;
        let n = 10_000;
        let mut rng = StdRng::seed_from_u64(0x2026_1031);
        let uniform = seeded_unit_vectors(n, dim, 0x2026_1031);
        let centres: Vec<Vec<f32>> = (0..16).map(|_| unit_gauss(&mut rng, dim)).collect();
        let clustered: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let c = &centres[rng.random_range(0..centres.len())];
                let mut v: Vec<f32> = c.iter().map(|x| x + 0.30 * gauss(&mut rng)).collect();
                let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect();

        for (name, vectors) in [("uniform", uniform), ("clustered16", clustered)] {
            let index = forest(&vectors, dim, 10, 98);
            let mut imbalances = Vec::new();
            let mut leaf_sizes = Vec::new();
            let mut depths = Vec::new();
            for tree in &index.trees {
                if let Some(root) = tree.root.as_ref() {
                    collect_split_stats(root, &mut imbalances, &mut leaf_sizes, &mut depths, 0);
                }
            }
            imbalances.sort_by(f32::total_cmp);
            let over = imbalances.iter().filter(|i| **i >= 0.95).count();
            let max_leaf = leaf_sizes.iter().copied().max().unwrap_or(0);
            let over_leaf = leaf_sizes.iter().filter(|l| **l > 98).count();
            let max_depth = depths.iter().map(|(d, _)| *d).max().unwrap_or(0);
            let mean = imbalances.iter().sum::<f32>() / imbalances.len().max(1) as f32;

            // How often the 0.95 guard has to reject an attempt: re-fit each
            // internal node's point set and count attempts to acceptance.
            let mut sets = Vec::new();
            for tree in &index.trees {
                if let Some(root) = tree.root.as_ref() {
                    internal_node_sets(root, &mut sets);
                }
            }
            let mut fit_rng = StdRng::seed_from_u64(0xBEEF);
            let mut accepted_at = [0usize; SPLIT_ATTEMPTS];
            let mut never_balanced = 0usize;
            for ids in &sets {
                let mut accepted = false;
                for slot in &mut accepted_at {
                    let Some(normal) = index.two_means_normal(ids, &mut fit_rng) else {
                        continue;
                    };
                    let (l, r) = index.partition_on(ids, &normal);
                    if split_imbalance(l.len(), r.len()) < SPLIT_MAX_IMBALANCE {
                        *slot += 1;
                        accepted = true;
                        break;
                    }
                }
                if !accepted {
                    never_balanced += 1;
                }
            }
            let attempts: Vec<String> = accepted_at.iter().map(usize::to_string).collect();

            println!(
                "{name}: splits={} imbalance mean={mean:.3} p50={:.3} p90={:.3} max={:.3} \
                 frac>=0.95={:.3} | leaves={} mean={:.1} max={max_leaf} over_cap={over_leaf} \
                 max_depth={max_depth} | refit accepted at attempt {} (1..3), never={never_balanced}",
                imbalances.len(),
                pct(&imbalances, 0.50),
                pct(&imbalances, 0.90),
                pct(&imbalances, 1.0),
                over as f32 / imbalances.len().max(1) as f32,
                leaf_sizes.len(),
                leaf_sizes.iter().sum::<usize>() as f32 / leaf_sizes.len().max(1) as f32,
                attempts.join("/"),
            );
        }
    }

    fn gauss(rng: &mut StdRng) -> f32 {
        let u1 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2 = rng.random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn unit_gauss(rng: &mut StdRng, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| gauss(rng)).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut v {
            *x /= norm;
        }
        v
    }

    #[test]
    fn search_routes_on_the_stored_threshold() {
        // Two one-point leaves on opposite sides of a plane whose threshold is
        // not zero. A budget of one leaf exposes which child the query took.
        let mut index = RpForestIndex::new(2, RpForestParams::default()).unwrap();
        index.add(0, vec![-1.0, 0.0]).unwrap();
        index.add(1, vec![0.0, 1.0]).unwrap();
        index.build().unwrap();
        index.trees = vec![RPTree {
            root: Some(TreeNode::Internal {
                hyperplane: vec![1.0, 0.0],
                threshold: 0.5,
                left: Box::new(TreeNode::Leaf { indices: vec![0] }),
                right: Box::new(TreeNode::Leaf { indices: vec![1] }),
            }),
        }];
        index.max_sq_norm = 1.0;
        // projection = 0.2, below the 0.5 threshold: the left leaf is near.
        let query = [0.2f32, 0.98];
        let split = index.search_with_budget(&query, 1, 1).unwrap();
        assert_eq!(
            split.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![0],
            "routed on 0.0 instead of the stored 0.5 threshold"
        );
        assert!((split[0].1 - 1.2).abs() < 1e-5);
        // Same tree, threshold moved to zero: the right leaf is now near.
        if let Some(TreeNode::Internal { threshold, .. }) = index.trees[0].root.as_mut() {
            *threshold = 0.0;
        }
        let zero = index.search_with_budget(&query, 1, 1).unwrap();
        assert_eq!(zero.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![1]);
    }
}
