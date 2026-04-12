//! Graph-based ANN with complex filter predicates (AND/OR/comparisons).
//! Related to ACORN (Patel et al., 2024) and Filtered-DiskANN.
//!
//! Combines a single-layer proximity graph (similar to NSG/PAG)
//! with per-attribute inverted indexes. At query time, the filter predicate is
//! resolved against the inverted indexes to obtain a matching ID set, and
//! search strategy is chosen based on selectivity:
//!
//! - **High selectivity** (>10% of corpus matches): beam search from medoid,
//!   post-filter candidates against the matching set.
//! - **Low selectivity** (<10% of corpus matches): brute-force scan over the
//!   matching IDs only.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.3", features = ["filtered_graph"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use std::collections::HashMap;
//! use vicinity::filtered_graph::{FilteredGraphIndex, FilteredGraphParams, AttrValue, Filter, Predicate};
//!
//! let params = FilteredGraphParams::default();
//! let mut index = FilteredGraphIndex::new(4, params)?;
//!
//! let mut attrs = HashMap::new();
//! attrs.insert("category".to_string(), AttrValue::String("news".to_string()));
//! attrs.insert("year".to_string(), AttrValue::Int(2023));
//! index.add(0, vec![1.0, 0.0, 0.0, 0.0], attrs)?;
//! index.build()?;
//!
//! let filter = Filter::Clause(Predicate::Eq("category".to_string(), AttrValue::String("news".to_string())));
//! let results = index.search_filtered(&[1.0, 0.0, 0.0, 0.0], 5, &filter)?;
//! ```
//!
//! # References
//!
//! - Gollapudi et al. (2023). "Filtered-DiskANN: Graph Algorithms for Approximate
//!   Nearest Neighbor Search with Filters." WWW 2023.
//! - Patel et al. (2024). "ACORN: Performant and Predicate-Agnostic Search Over
//!   Vector Embeddings and Structured Data." arXiv:2403.04871.

use crate::distance::cosine_distance_normalized;
use crate::RetrieveError;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

// ── Public types ──────────────────────────────────────────────────────────────

/// A typed attribute value stored alongside each vector.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    /// UTF-8 string.
    String(String),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit float.
    Float(f64),
    /// Boolean.
    Bool(bool),
}

/// A single comparison predicate over one attribute.
#[derive(Clone, Debug)]
pub enum Predicate {
    /// Attribute equals value.
    Eq(String, AttrValue),
    /// Attribute strictly less than value.
    Lt(String, AttrValue),
    /// Attribute less than or equal to value.
    Le(String, AttrValue),
    /// Attribute strictly greater than value.
    Gt(String, AttrValue),
    /// Attribute greater than or equal to value.
    Ge(String, AttrValue),
    /// Attribute value is one of the provided list.
    In(String, Vec<AttrValue>),
}

/// A boolean filter tree over [`Predicate`]s.
#[derive(Clone, Debug)]
pub enum Filter {
    /// A single predicate leaf.
    Clause(Predicate),
    /// All child filters must match (AND).
    And(Vec<Filter>),
    /// At least one child filter must match (OR).
    Or(Vec<Filter>),
}

// ── Parameters ────────────────────────────────────────────────────────────────

/// Construction and search parameters for [`FilteredGraphIndex`].
#[derive(Clone, Debug)]
pub struct FilteredGraphParams {
    /// Maximum out-degree in the proximity graph. Default: 32.
    pub max_degree: usize,
    /// Candidate pool size during graph construction. Default: 200.
    pub ef_construction: usize,
    /// Candidate pool size during search. Default: 100.
    pub ef_search: usize,
    /// RNG pruning threshold (alpha > 1 relaxes diversity). Default: 1.2.
    pub alpha: f32,
}

impl Default for FilteredGraphParams {
    fn default() -> Self {
        Self {
            max_degree: 32,
            ef_construction: 200,
            ef_search: 100,
            alpha: 1.2,
        }
    }
}

// ── Index ─────────────────────────────────────────────────────────────────────

/// FilteredGraph index: proximity graph + per-attribute inverted indexes.
pub struct FilteredGraphIndex {
    dimension: usize,
    params: FilteredGraphParams,
    built: bool,

    /// Flat storage: internal index i -> vectors[i*dim .. (i+1)*dim].
    vectors: Vec<f32>,
    num_vectors: usize,

    /// External doc IDs parallel to vector storage.
    doc_ids: Vec<u32>,

    /// Attribute maps during the staging phase (cleared after build).
    staging_attrs: Vec<HashMap<String, AttrValue>>,

    /// Adjacency lists (internal IDs).
    neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Entry point: internal ID of the medoid.
    medoid: u32,

    /// Per-attribute inverted index.
    ///
    /// `inverted[attr]` is a sorted list of `(value, sorted_internal_ids)`.
    /// Sorted by value so range lookups can binary-search.
    inverted: HashMap<String, Vec<(AttrValue, Vec<u32>)>>,
}

impl FilteredGraphIndex {
    /// Create a new (empty, unbuilt) index.
    ///
    /// Returns an error when `dimension == 0`.
    pub fn new(dimension: usize, params: FilteredGraphParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::InvalidParameter(
                "dimension must be > 0".into(),
            ));
        }
        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            doc_ids: Vec::new(),
            staging_attrs: Vec::new(),
            neighbors: Vec::new(),
            medoid: 0,
            inverted: HashMap::new(),
        })
    }

    /// Add a vector with associated attributes.
    pub fn add(
        &mut self,
        doc_id: u32,
        vector: Vec<f32>,
        attrs: HashMap<String, AttrValue>,
    ) -> Result<(), RetrieveError> {
        self.add_slice(doc_id, &vector, attrs)
    }

    /// Add a vector from a slice with associated attributes.
    pub fn add_slice(
        &mut self,
        doc_id: u32,
        vector: &[f32],
        attrs: HashMap<String, AttrValue>,
    ) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add vectors after build".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            self.vectors.extend(vector.iter().map(|x| x / norm));
        } else {
            self.vectors.extend_from_slice(vector);
        }
        self.doc_ids.push(doc_id);
        self.staging_attrs.push(attrs);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the graph and inverted indexes.
    ///
    /// Must be called before any search. Calling `build` a second time is a no-op.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        self.medoid = self.compute_medoid();
        self.build_knn_graph();
        self.rng_refine();
        self.ensure_connectivity();
        self.build_inverted_indexes();

        self.built = true;
        Ok(())
    }

    /// Search for the k nearest neighbors of `query` that satisfy `filter`.
    ///
    /// The strategy (graph beam search with post-filter vs brute-force over
    /// matching IDs) is chosen automatically based on selectivity.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let qn = normalize(query);
        let matching = self.evaluate_filter(filter);

        if matching.is_empty() {
            return Ok(Vec::new());
        }

        let selectivity = matching.len() as f32 / self.num_vectors as f32;

        let mut results: Vec<(u32, f32)> = if selectivity >= 0.10 {
            // High selectivity: beam search, then retain only matching IDs.
            let ef = self.params.ef_search.max(k * 4);
            let candidates = self.beam_search(&qn, ef);
            candidates
                .into_iter()
                .filter(|(id, _)| matching.contains(id))
                .take(k)
                .map(|(id, d)| (self.doc_ids[id as usize], d))
                .collect()
        } else {
            // Low selectivity: brute-force over matching IDs only.
            let mut scored: Vec<(u32, f32)> = matching
                .iter()
                .map(|&id| {
                    let d = cosine_distance_normalized(&qn, self.get_vector(id as usize));
                    (id, d)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            scored.truncate(k);
            scored
                .into_iter()
                .map(|(id, d)| (self.doc_ids[id as usize], d))
                .collect()
        };

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(results)
    }

    /// Unfiltered nearest-neighbor search.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let qn = normalize(query);
        let ef = self.params.ef_search.max(k);
        let candidates = self.beam_search(&qn, ef);

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(id, d)| (self.doc_ids[id as usize], d))
            .collect())
    }

    /// Unfiltered search with a custom `ef_search` beam width.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::InvalidParameter(
                "index must be built before search".into(),
            ));
        }
        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: query.len(),
                doc_dim: self.dimension,
            });
        }

        let qn = normalize(query);
        let ef = ef_search.max(k);
        let candidates = self.beam_search(&qn, ef);

        Ok(candidates
            .into_iter()
            .take(k)
            .map(|(id, d)| (self.doc_ids[id as usize], d))
            .collect())
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index holds no vectors.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    // ── Filter evaluation ─────────────────────────────────────────────────────

    /// Evaluate `filter` against the inverted indexes and return the set of
    /// matching *internal* IDs.
    fn evaluate_filter(&self, filter: &Filter) -> HashSet<u32> {
        match filter {
            Filter::Clause(pred) => self.evaluate_predicate(pred),
            Filter::And(children) => {
                if children.is_empty() {
                    // Vacuously true: all IDs match.
                    return (0..self.num_vectors as u32).collect();
                }
                let mut result = self.evaluate_filter(&children[0]);
                for child in &children[1..] {
                    if result.is_empty() {
                        break;
                    }
                    let child_set = self.evaluate_filter(child);
                    result.retain(|id| child_set.contains(id));
                }
                result
            }
            Filter::Or(children) => {
                let mut result = HashSet::new();
                for child in children {
                    result.extend(self.evaluate_filter(child));
                }
                result
            }
        }
    }

    fn evaluate_predicate(&self, pred: &Predicate) -> HashSet<u32> {
        match pred {
            Predicate::Eq(attr, val) => self.inverted_eq(attr, val),
            Predicate::Lt(attr, val) => self.inverted_range(attr, None, Some(val), false, false),
            Predicate::Le(attr, val) => self.inverted_range(attr, None, Some(val), false, true),
            Predicate::Gt(attr, val) => self.inverted_range(attr, Some(val), None, false, false),
            Predicate::Ge(attr, val) => self.inverted_range(attr, Some(val), None, true, false),
            Predicate::In(attr, vals) => {
                let mut result = HashSet::new();
                for v in vals {
                    result.extend(self.inverted_eq(attr, v));
                }
                result
            }
        }
    }

    /// Return internal IDs where `attr == val`.
    fn inverted_eq(&self, attr: &str, val: &AttrValue) -> HashSet<u32> {
        let Some(entries) = self.inverted.get(attr) else {
            return HashSet::new();
        };
        // Binary search for matching value.
        let pos = entries
            .partition_point(|(v, _)| compare_attr(v, val).map_or(true, |o| o == Ordering::Less));
        let mut result = HashSet::new();
        for (v, ids) in &entries[pos..] {
            if compare_attr(v, val) != Some(Ordering::Equal) {
                break;
            }
            result.extend(ids.iter().copied());
        }
        result
    }

    /// Return internal IDs satisfying a range constraint.
    ///
    /// `lo` is a lower bound; `hi` is an upper bound. `lo_inclusive` and
    /// `hi_inclusive` control whether the bounds are included.
    fn inverted_range(
        &self,
        attr: &str,
        lo: Option<&AttrValue>,
        hi: Option<&AttrValue>,
        lo_inclusive: bool,
        hi_inclusive: bool,
    ) -> HashSet<u32> {
        let Some(entries) = self.inverted.get(attr) else {
            return HashSet::new();
        };

        let mut result = HashSet::new();
        for (v, ids) in entries {
            let lo_ok = match lo {
                None => true,
                Some(bound) => match compare_attr(v, bound) {
                    Some(Ordering::Greater) => true,
                    Some(Ordering::Equal) => lo_inclusive,
                    _ => false,
                },
            };
            let hi_ok = match hi {
                None => true,
                Some(bound) => match compare_attr(v, bound) {
                    Some(Ordering::Less) => true,
                    Some(Ordering::Equal) => hi_inclusive,
                    _ => false,
                },
            };
            if lo_ok && hi_ok {
                result.extend(ids.iter().copied());
            }
        }
        result
    }

    // ── Graph construction ────────────────────────────────────────────────────

    fn compute_medoid(&self) -> u32 {
        let dim = self.dimension;
        let n = self.num_vectors;
        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            let v = self.get_vector(i);
            for (j, &val) in v.iter().enumerate() {
                centroid[j] += val;
            }
        }
        for c in &mut centroid {
            *c /= n as f32;
        }
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for i in 0..n {
            let d = cosine_distance_normalized(&centroid, self.get_vector(i));
            if d < best_d {
                best_d = d;
                best = i as u32;
            }
        }
        best
    }

    /// Build initial kNN graph. Brute-force for n <= 1000, NN-descent otherwise.
    fn build_knn_graph(&mut self) {
        let n = self.num_vectors;
        if n <= 1000 {
            self.build_knn_graph_bruteforce();
        } else {
            self.build_knn_graph_nndescent();
        }
    }

    fn build_knn_graph_bruteforce(&mut self) {
        let n = self.num_vectors;
        let k = self.params.max_degree.min(n.saturating_sub(1));
        self.neighbors = vec![SmallVec::new(); n];

        for i in 0..n {
            let vi = self.get_vector(i);
            let mut dists: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, cosine_distance_normalized(vi, self.get_vector(j))))
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(k);
            self.neighbors[i] = dists.iter().map(|(id, _)| *id).collect();
        }
    }

    /// NN-descent kNN graph construction.
    fn build_knn_graph_nndescent(&mut self) {
        let (n, k, dim) = (self.num_vectors, self.params.max_degree, self.dimension);
        let vecs = &self.vectors;
        self.neighbors = crate::graph_utils::build_knn_graph_nndescent(n, k, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }

    /// RNG refinement pass: beam search for candidates, then alpha-pruning.
    fn rng_refine(&mut self) {
        let n = self.num_vectors;
        let ef = self.params.ef_construction;

        for i in 0..n {
            let vi = self.get_vector(i).to_vec();
            let candidates = self.beam_search(&vi, ef);
            let selected = self.rng_prune(&vi, &candidates);

            let old = std::mem::replace(
                &mut self.neighbors[i],
                selected.iter().map(|&(id, _)| id).collect(),
            );

            let max_deg = self.params.max_degree;
            for &(nb_id, _) in &selected {
                let nid = nb_id as usize;
                if !self.neighbors[nid].contains(&(i as u32)) {
                    if self.neighbors[nid].len() < max_deg {
                        self.neighbors[nid].push(i as u32);
                    } else {
                        let nv = self.get_vector(nid).to_vec();
                        let rev_cands: Vec<(u32, f32)> = self.neighbors[nid]
                            .iter()
                            .chain(std::iter::once(&(i as u32)))
                            .map(|&id| {
                                let d =
                                    cosine_distance_normalized(&nv, self.get_vector(id as usize));
                                (id, d)
                            })
                            .collect();
                        let pruned = self.rng_prune(&nv, &rev_cands);
                        self.neighbors[nid] = pruned.iter().map(|&(id, _)| id).collect();
                    }
                }
            }

            drop(old);
        }
    }

    /// Alpha-RNG pruning: keep candidate if no already-selected neighbor is
    /// within `alpha * dist(query, candidate)` of the candidate.
    fn rng_prune(&self, _query_vec: &[f32], candidates: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let mut sorted: Vec<(u32, f32)> = candidates.to_vec();
        sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        sorted.dedup_by_key(|c| c.0);

        let max_deg = self.params.max_degree;
        let alpha = self.params.alpha;
        let mut selected: Vec<(u32, f32)> = Vec::with_capacity(max_deg);

        'outer: for &(cand_id, cand_dist) in &sorted {
            if selected.len() >= max_deg {
                break;
            }
            let cand_vec = self.get_vector(cand_id as usize);
            for &(sel_id, _) in &selected {
                let sel_vec = self.get_vector(sel_id as usize);
                let inter = cosine_distance_normalized(sel_vec, cand_vec);
                // Discard candidate if a selected neighbor is already within
                // alpha * cand_dist of it (Vamana/DiskANN RNG condition).
                if alpha * cand_dist >= inter {
                    continue 'outer;
                }
            }
            selected.push((cand_id, cand_dist));
        }

        selected
    }

    fn ensure_connectivity(&mut self) {
        let (dim, vecs) = (self.dimension, &self.vectors);
        crate::graph_utils::ensure_connectivity(&mut self.neighbors, self.medoid, |i, j| {
            cosine_distance_normalized(&vecs[i * dim..(i + 1) * dim], &vecs[j * dim..(j + 1) * dim])
        });
    }

    // ── Inverted index construction ───────────────────────────────────────────

    fn build_inverted_indexes(&mut self) {
        // Collect (attr, value, internal_id) triples.
        let mut raw: HashMap<String, Vec<(AttrValue, u32)>> = HashMap::new();

        for (internal_id, attrs) in self.staging_attrs.iter().enumerate() {
            for (attr, val) in attrs {
                raw.entry(attr.clone())
                    .or_default()
                    .push((val.clone(), internal_id as u32));
            }
        }

        // Group by value: for each attribute, build sorted list of (value, ids).
        for (attr, mut pairs) in raw {
            // Sort by value for binary-searchable range lookups.
            pairs.sort_unstable_by(|(a, _), (b, _)| compare_attr(a, b).unwrap_or(Ordering::Equal));

            // Collapse consecutive equal values into one entry.
            let mut entries: Vec<(AttrValue, Vec<u32>)> = Vec::new();
            for (val, id) in pairs {
                if let Some(last) = entries.last_mut() {
                    if last.0 == val {
                        last.1.push(id);
                        continue;
                    }
                }
                entries.push((val, vec![id]));
            }

            // Sort IDs within each bucket for stable output.
            for (_, ids) in &mut entries {
                ids.sort_unstable();
            }

            self.inverted.insert(attr, entries);
        }

        // Free staging memory.
        self.staging_attrs = Vec::new();
    }

    // ── Beam search ───────────────────────────────────────────────────────────

    fn beam_search(&self, query: &[f32], ef: usize) -> Vec<(u32, f32)> {
        let n = self.num_vectors;
        if n == 0 {
            return Vec::new();
        }

        let mut visited: HashSet<u32> = HashSet::new();
        let mut frontier: BinaryHeap<std::cmp::Reverse<(FloatOrd, u32)>> = BinaryHeap::new();
        let mut candidates: Vec<(u32, f32)> = Vec::new();

        let entry = self.medoid;
        let entry_dist = cosine_distance_normalized(query, self.get_vector(entry as usize));
        visited.insert(entry);
        frontier.push(std::cmp::Reverse((FloatOrd(entry_dist), entry)));
        candidates.push((entry, entry_dist));

        while let Some(std::cmp::Reverse((FloatOrd(current_dist), current_id))) = frontier.pop() {
            // Early exit: if we already have ef candidates and current is worse
            // than the ef-th, no neighbor can improve the result set.
            if candidates.len() >= ef {
                candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                if current_dist > candidates[ef - 1].1 * 1.5 {
                    break;
                }
            }

            for &nb in &self.neighbors[current_id as usize] {
                if visited.insert(nb) {
                    let d = cosine_distance_normalized(query, self.get_vector(nb as usize));
                    candidates.push((nb, d));
                    frontier.push(std::cmp::Reverse((FloatOrd(d), nb)));
                }
            }

            if visited.len() > ef * 10 {
                break;
            }
        }

        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.dedup_by_key(|c| c.0);
        candidates
    }

    // ── Vector access ─────────────────────────────────────────────────────────

    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Compare two [`AttrValue`]s of the same variant. Returns `None` for
/// cross-type comparisons.
fn compare_attr(a: &AttrValue, b: &AttrValue) -> Option<Ordering> {
    match (a, b) {
        (AttrValue::Int(x), AttrValue::Int(y)) => Some(x.cmp(y)),
        (AttrValue::Float(x), AttrValue::Float(y)) => x.partial_cmp(y),
        (AttrValue::String(x), AttrValue::String(y)) => Some(x.cmp(y)),
        (AttrValue::Bool(x), AttrValue::Bool(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

// ── FloatOrd ──────────────────────────────────────────────────────────────────

use crate::distance::FloatOrd;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n * dim).map(|_| lcg(&mut s)).collect()
    }

    fn build_index(n: usize, dim: usize, seed: u64) -> FilteredGraphIndex {
        let data = make_vectors(n, dim, seed);
        let params = FilteredGraphParams {
            max_degree: 16,
            ef_construction: 40,
            ef_search: 40,
            alpha: 1.2,
        };
        let mut idx = FilteredGraphIndex::new(dim, params).unwrap();
        for i in 0..n {
            let start = i * dim;
            idx.add_slice(i as u32, &data[start..start + dim], HashMap::new())
                .unwrap();
        }
        idx.build().unwrap();
        idx
    }

    // ── Unfiltered search ────────────────────────────────────────────────────

    #[test]
    fn build_and_search_unfiltered() {
        let dim = 16;
        let n = 50;
        let idx = build_index(n, dim, 42);

        let data = make_vectors(n, dim, 42);
        let query = &data[0..dim];
        let results = idx.search(query, 5).unwrap();

        assert!(!results.is_empty());
        // Self should be the nearest neighbor.
        assert!(results.iter().any(|(id, _)| *id == 0));
    }

    // ── Equality filter ───────────────────────────────────────────────────────

    #[test]
    fn eq_filter() {
        let dim = 8;
        let n = 30;
        let data = make_vectors(n, dim, 7);
        let params = FilteredGraphParams {
            max_degree: 12,
            ef_construction: 30,
            ef_search: 30,
            alpha: 1.2,
        };
        let mut idx = FilteredGraphIndex::new(dim, params).unwrap();

        for i in 0..n {
            let start = i * dim;
            let label = if i % 2 == 0 { "even" } else { "odd" };
            let mut attrs = HashMap::new();
            attrs.insert("parity".to_string(), AttrValue::String(label.to_string()));
            idx.add_slice(i as u32, &data[start..start + dim], attrs)
                .unwrap();
        }
        idx.build().unwrap();

        let filter = Filter::Clause(Predicate::Eq(
            "parity".to_string(),
            AttrValue::String("even".to_string()),
        ));
        let results = idx.search_filtered(&data[0..dim], 20, &filter).unwrap();

        // All returned IDs must be even.
        for (doc_id, _) in &results {
            assert_eq!(doc_id % 2, 0, "doc_id {doc_id} is not even");
        }
        assert!(!results.is_empty());
    }

    // ── Range filter (Lt / Gt on Int) ─────────────────────────────────────────

    #[test]
    fn range_filter() {
        let dim = 8;
        let n = 40;
        let data = make_vectors(n, dim, 13);
        let params = FilteredGraphParams {
            max_degree: 12,
            ef_construction: 30,
            ef_search: 30,
            alpha: 1.2,
        };
        let mut idx = FilteredGraphIndex::new(dim, params).unwrap();

        for i in 0..n {
            let start = i * dim;
            let mut attrs = HashMap::new();
            attrs.insert("score".to_string(), AttrValue::Int(i as i64));
            idx.add_slice(i as u32, &data[start..start + dim], attrs)
                .unwrap();
        }
        idx.build().unwrap();

        // Request docs with score in (9, 21): Gt(9) AND Lt(21).
        let filter = Filter::And(vec![
            Filter::Clause(Predicate::Gt("score".to_string(), AttrValue::Int(9))),
            Filter::Clause(Predicate::Lt("score".to_string(), AttrValue::Int(21))),
        ]);
        let results = idx.search_filtered(&data[0..dim], 20, &filter).unwrap();

        for (doc_id, _) in &results {
            let score = *doc_id as i64; // doc_id == i == score in this test
            assert!(score > 9 && score < 21, "score {score} outside (9, 21)");
        }
        assert!(!results.is_empty());
    }

    // ── AND filter ────────────────────────────────────────────────────────────

    #[test]
    fn and_filter() {
        let dim = 8;
        let n = 40;
        let data = make_vectors(n, dim, 17);
        let params = FilteredGraphParams {
            max_degree: 12,
            ef_construction: 30,
            ef_search: 30,
            alpha: 1.2,
        };
        let mut idx = FilteredGraphIndex::new(dim, params).unwrap();

        for i in 0..n {
            let start = i * dim;
            let mut attrs = HashMap::new();
            let parity = if i % 2 == 0 { "even" } else { "odd" };
            let tier = if i < 20 { "low" } else { "high" };
            attrs.insert("parity".to_string(), AttrValue::String(parity.to_string()));
            attrs.insert("tier".to_string(), AttrValue::String(tier.to_string()));
            idx.add_slice(i as u32, &data[start..start + dim], attrs)
                .unwrap();
        }
        idx.build().unwrap();

        // Even AND high tier (i >= 20 and i % 2 == 0 -> i in {20,22,24,...,38})
        let filter = Filter::And(vec![
            Filter::Clause(Predicate::Eq(
                "parity".to_string(),
                AttrValue::String("even".to_string()),
            )),
            Filter::Clause(Predicate::Eq(
                "tier".to_string(),
                AttrValue::String("high".to_string()),
            )),
        ]);
        let results = idx.search_filtered(&data[0..dim], 20, &filter).unwrap();

        for (doc_id, _) in &results {
            assert_eq!(doc_id % 2, 0, "doc_id {doc_id} should be even");
            assert!(*doc_id >= 20, "doc_id {doc_id} should be >= 20");
        }
    }

    // ── OR filter ─────────────────────────────────────────────────────────────

    #[test]
    fn or_filter() {
        let dim = 8;
        let n = 30;
        let data = make_vectors(n, dim, 23);
        let params = FilteredGraphParams {
            max_degree: 12,
            ef_construction: 30,
            ef_search: 30,
            alpha: 1.2,
        };
        let mut idx = FilteredGraphIndex::new(dim, params).unwrap();

        for i in 0..n {
            let start = i * dim;
            let mut attrs = HashMap::new();
            let label = match i % 3 {
                0 => "alpha",
                1 => "beta",
                _ => "gamma",
            };
            attrs.insert("group".to_string(), AttrValue::String(label.to_string()));
            idx.add_slice(i as u32, &data[start..start + dim], attrs)
                .unwrap();
        }
        idx.build().unwrap();

        // Group "alpha" OR "beta" -- must exclude "gamma".
        let filter = Filter::Or(vec![
            Filter::Clause(Predicate::Eq(
                "group".to_string(),
                AttrValue::String("alpha".to_string()),
            )),
            Filter::Clause(Predicate::Eq(
                "group".to_string(),
                AttrValue::String("beta".to_string()),
            )),
        ]);
        let results = idx.search_filtered(&data[0..dim], 20, &filter).unwrap();

        for (doc_id, _) in &results {
            assert_ne!(
                doc_id % 3,
                2,
                "doc_id {doc_id} should not be in group gamma"
            );
        }
        assert!(!results.is_empty());
    }

    // ── No-match returns empty ────────────────────────────────────────────────

    #[test]
    fn no_match_returns_empty() {
        let dim = 8;
        let n = 20;
        let data = make_vectors(n, dim, 99);
        let params = FilteredGraphParams {
            max_degree: 8,
            ef_construction: 20,
            ef_search: 20,
            alpha: 1.2,
        };
        let mut idx = FilteredGraphIndex::new(dim, params).unwrap();
        for i in 0..n {
            let start = i * dim;
            let mut attrs = HashMap::new();
            attrs.insert("tag".to_string(), AttrValue::String("present".to_string()));
            idx.add_slice(i as u32, &data[start..start + dim], attrs)
                .unwrap();
        }
        idx.build().unwrap();

        let filter = Filter::Clause(Predicate::Eq(
            "tag".to_string(),
            AttrValue::String("absent".to_string()),
        ));
        let results = idx.search_filtered(&data[0..dim], 5, &filter).unwrap();
        assert!(results.is_empty());
    }

    // ── Empty index errors ────────────────────────────────────────────────────

    #[test]
    fn empty_index_errors() {
        let mut idx = FilteredGraphIndex::new(8, FilteredGraphParams::default()).unwrap();
        assert!(idx.build().is_err());
    }
}
