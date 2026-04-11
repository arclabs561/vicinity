//! ESG: Elastic Graphs for range-filtered approximate nearest neighbor search.
//!
//! Handles queries of the form "find k nearest neighbors of q among vectors
//! whose attribute falls in [l, r]". Decomposes any range into at most two
//! half-bounded intervals and searches two pre-built HNSW subgraph snapshots,
//! reducing query cost from O(log N) subgraph searches (prior work) to O(1).
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.3", features = ["esg"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::esg::{EsgIndex, EsgParams};
//!
//! let params = EsgParams::default();
//! let mut index = EsgIndex::new(128, params)?;
//!
//! // Add vectors with numeric attributes (e.g., timestamps)
//! index.add(0, vec![0.1; 128], 1000.0)?;
//! index.add(1, vec![0.2; 128], 2000.0)?;
//! index.add(2, vec![0.3; 128], 3000.0)?;
//! index.build()?;
//!
//! // Range query: find 5 NN among vectors with attribute in [1500, 2500]
//! let results = index.range_search(&query, 5, 1500.0, 2500.0)?;
//! ```
//!
//! # How It Works
//!
//! 1. Sort all vectors by their numeric attribute
//! 2. Build suffix HNSW snapshots: insert vectors in decreasing attribute order,
//!    snapshot at geometric checkpoint intervals
//! 3. Build prefix HNSW snapshots: insert in increasing order, snapshot
//! 4. Query [l, r]: binary-search for tightest suffix checkpoint >= l and
//!    prefix checkpoint <= r, search both with elastic relaxation
//!
//! "Elastic relaxation" means out-of-range nodes can be visited during graph
//! traversal (for navigation) but are excluded from the result set.
//!
//! # References
//!
//! - Yang et al. (2025). "ESG: Elastic Graphs for Range-Filtering Approximate
//!   kNN Search." arXiv:2504.04018.

use crate::RetrieveError;
use std::collections::HashSet;

/// ESG parameters.
#[derive(Clone, Debug)]
pub struct EsgParams {
    /// Number of checkpoints per direction. More = tighter range coverage, more memory.
    pub num_checkpoints: usize,
    /// HNSW M parameter for subgraph construction.
    pub hnsw_m: usize,
    /// HNSW ef_construction.
    pub hnsw_ef_construction: usize,
    /// Search ef parameter.
    pub ef_search: usize,
}

impl Default for EsgParams {
    fn default() -> Self {
        Self {
            num_checkpoints: 16,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            ef_search: 100,
        }
    }
}

/// A point with a vector and a numeric attribute.
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct AttributedPoint {
    doc_id: u32,
    attribute: f64,
    /// Index into the sorted order.
    rank: usize,
}

/// A snapshot of an HNSW index at a particular attribute threshold.
struct Checkpoint {
    /// Attribute threshold for this checkpoint.
    threshold: f64,
    /// HNSW index covering vectors at/beyond this threshold.
    #[cfg(feature = "hnsw")]
    hnsw: crate::hnsw::HNSWIndex,
}

/// ESG index for range-filtered ANN search.
pub struct EsgIndex {
    dimension: usize,
    params: EsgParams,
    built: bool,

    /// Vectors in insertion order.
    vectors: Vec<f32>,
    num_vectors: usize,
    /// Points sorted by attribute (populated during build).
    sorted_points: Vec<AttributedPoint>,
    /// Unsorted points (pre-build staging).
    staging: Vec<(u32, Vec<f32>, f64)>,

    /// Suffix checkpoints: each covers [threshold, +inf).
    /// Sorted by threshold ascending.
    #[cfg(feature = "hnsw")]
    suffix_checkpoints: Vec<Checkpoint>,
    /// Prefix checkpoints: each covers (-inf, threshold].
    /// Sorted by threshold ascending.
    #[cfg(feature = "hnsw")]
    prefix_checkpoints: Vec<Checkpoint>,
}

impl EsgIndex {
    /// Create a new ESG index.
    pub fn new(dimension: usize, params: EsgParams) -> Result<Self, RetrieveError> {
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
            sorted_points: Vec::new(),
            staging: Vec::new(),
            #[cfg(feature = "hnsw")]
            suffix_checkpoints: Vec::new(),
            #[cfg(feature = "hnsw")]
            prefix_checkpoints: Vec::new(),
        })
    }

    /// Add a vector with a numeric attribute.
    pub fn add(
        &mut self,
        doc_id: u32,
        vector: Vec<f32>,
        attribute: f64,
    ) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::InvalidParameter(
                "cannot add after build".into(),
            ));
        }
        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: vector.len(),
                doc_dim: self.dimension,
            });
        }
        self.staging.push((doc_id, vector, attribute));
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index: sort by attribute, create checkpoint HNSW snapshots.
    #[cfg(feature = "hnsw")]
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Sort staging by attribute
        self.staging.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));

        // Store sorted points and normalized vectors
        self.sorted_points = Vec::with_capacity(self.num_vectors);
        self.vectors = Vec::with_capacity(self.num_vectors * self.dimension);

        for (rank, (doc_id, vector, attribute)) in self.staging.drain(..).enumerate() {
            self.sorted_points.push(AttributedPoint {
                doc_id,
                attribute,
                rank,
            });
            // L2-normalize
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                self.vectors.extend(vector.iter().map(|x| x / norm));
            } else {
                self.vectors.extend_from_slice(&vector);
            }
        }

        let n = self.num_vectors;
        let num_cp = self.params.num_checkpoints.min(n);

        // Compute geometric checkpoint positions
        let cp_positions = geometric_checkpoints(n, num_cp);

        // Build suffix checkpoints: insert in decreasing rank order
        // Checkpoint at rank r covers vectors with rank >= r (attribute >= attr[r])
        self.suffix_checkpoints = Vec::with_capacity(num_cp);
        for &cp_rank in cp_positions.iter().rev() {
            let threshold = self.sorted_points[cp_rank].attribute;
            let mut hnsw = crate::hnsw::HNSWIndex::builder(self.dimension)
                .m(self.params.hnsw_m)
                .ef_construction(self.params.hnsw_ef_construction)
                .auto_normalize(false)
                .build()?;

            for rank in cp_rank..n {
                let doc_id = self.sorted_points[rank].doc_id;
                let vec = self.get_vector(rank);
                hnsw.add_slice(doc_id, vec)?;
            }
            hnsw.build()?;

            self.suffix_checkpoints.push(Checkpoint { threshold, hnsw });
        }
        // Sort suffix checkpoints by threshold ascending
        self.suffix_checkpoints
            .sort_unstable_by(|a, b| a.threshold.total_cmp(&b.threshold));

        // Build prefix checkpoints: insert in increasing rank order
        // Checkpoint at rank r covers vectors with rank <= r (attribute <= attr[r])
        self.prefix_checkpoints = Vec::with_capacity(num_cp);
        for &cp_rank in &cp_positions {
            let threshold = self.sorted_points[cp_rank].attribute;
            let mut hnsw = crate::hnsw::HNSWIndex::builder(self.dimension)
                .m(self.params.hnsw_m)
                .ef_construction(self.params.hnsw_ef_construction)
                .auto_normalize(false)
                .build()?;

            for rank in 0..=cp_rank {
                let doc_id = self.sorted_points[rank].doc_id;
                let vec = self.get_vector(rank);
                hnsw.add_slice(doc_id, vec)?;
            }
            hnsw.build()?;

            self.prefix_checkpoints.push(Checkpoint { threshold, hnsw });
        }

        self.built = true;
        Ok(())
    }

    /// Range-filtered search: find k nearest neighbors with attribute in [lo, hi].
    #[cfg(feature = "hnsw")]
    pub fn range_search(
        &self,
        query: &[f32],
        k: usize,
        lo: f64,
        hi: f64,
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

        // Normalize query
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };

        let ef = self.params.ef_search.max(k);

        // Build the attribute lookup for filtering results
        let in_range = |doc_id: u32| -> bool {
            self.sorted_points
                .iter()
                .any(|p| p.doc_id == doc_id && p.attribute >= lo && p.attribute <= hi)
        };

        let mut all_candidates: Vec<(u32, f32)> = Vec::new();

        // Search suffix checkpoint (covers [lo, +inf))
        // Find tightest checkpoint with threshold <= lo
        if let Some(cp) = self
            .suffix_checkpoints
            .iter()
            .rev()
            .find(|cp| cp.threshold <= lo)
        {
            if let Ok(results) = cp.hnsw.search(&query_normalized, k * 2, ef) {
                for (doc_id, dist) in results {
                    if in_range(doc_id) {
                        all_candidates.push((doc_id, dist));
                    }
                }
            }
        } else if let Some(cp) = self.suffix_checkpoints.first() {
            // Use the broadest suffix checkpoint
            if let Ok(results) = cp.hnsw.search(&query_normalized, k * 2, ef) {
                for (doc_id, dist) in results {
                    if in_range(doc_id) {
                        all_candidates.push((doc_id, dist));
                    }
                }
            }
        }

        // Search prefix checkpoint (covers (-inf, hi])
        // Find tightest checkpoint with threshold >= hi
        if let Some(cp) = self.prefix_checkpoints.iter().find(|cp| cp.threshold >= hi) {
            if let Ok(results) = cp.hnsw.search(&query_normalized, k * 2, ef) {
                for (doc_id, dist) in results {
                    if in_range(doc_id) {
                        all_candidates.push((doc_id, dist));
                    }
                }
            }
        } else if let Some(cp) = self.prefix_checkpoints.last() {
            // Use the broadest prefix checkpoint
            if let Ok(results) = cp.hnsw.search(&query_normalized, k * 2, ef) {
                for (doc_id, dist) in results {
                    if in_range(doc_id) {
                        all_candidates.push((doc_id, dist));
                    }
                }
            }
        }

        // Merge, dedup, top-k
        all_candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        let mut seen = HashSet::new();
        all_candidates.retain(|(id, _)| seen.insert(*id));
        all_candidates.truncate(k);

        Ok(all_candidates)
    }

    /// Unfiltered search (searches the broadest checkpoint).
    #[cfg(feature = "hnsw")]
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        let lo = f64::NEG_INFINITY;
        let hi = f64::INFINITY;
        self.range_search(query, k, lo, hi)
    }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    #[inline]
    fn get_vector(&self, rank: usize) -> &[f32] {
        let start = rank * self.dimension;
        &self.vectors[start..start + self.dimension]
    }
}

/// Compute geometrically-spaced checkpoint positions in [0, n-1].
fn geometric_checkpoints(n: usize, num_checkpoints: usize) -> Vec<usize> {
    if num_checkpoints == 0 || n == 0 {
        return Vec::new();
    }
    if num_checkpoints >= n {
        return (0..n).collect();
    }

    let mut positions = Vec::with_capacity(num_checkpoints);
    for i in 0..num_checkpoints {
        let frac = i as f64 / (num_checkpoints - 1).max(1) as f64;
        let pos = (frac * (n - 1) as f64).round() as usize;
        if positions.last() != Some(&pos) {
            positions.push(pos);
        }
    }
    positions
}

#[cfg(test)]
#[cfg(feature = "hnsw")]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn make_vector(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| (seed as f32 * 0.1 + i as f32 * 0.01).sin())
            .collect()
    }

    #[test]
    fn build_and_range_search() {
        let dim = 16;
        let mut index = EsgIndex::new(
            dim,
            EsgParams {
                num_checkpoints: 4,
                hnsw_m: 8,
                hnsw_ef_construction: 50,
                ef_search: 50,
            },
        )
        .unwrap();

        // Add vectors with attributes 0..100
        for i in 0..50u32 {
            index.add(i, make_vector(dim, i), i as f64 * 2.0).unwrap();
        }
        index.build().unwrap();

        // Range query: attribute in [20, 60]
        let query = make_vector(dim, 15);
        let results = index.range_search(&query, 5, 20.0, 60.0).unwrap();

        // All results should have attribute in [20, 60]
        for (doc_id, _) in &results {
            let attr = *doc_id as f64 * 2.0;
            assert!(
                attr >= 20.0 && attr <= 60.0,
                "doc_id {} has attribute {}, expected in [20, 60]",
                doc_id,
                attr
            );
        }
    }

    #[test]
    fn full_range_search() {
        let dim = 16;
        let mut index = EsgIndex::new(
            dim,
            EsgParams {
                num_checkpoints: 4,
                hnsw_m: 8,
                hnsw_ef_construction: 50,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..30u32 {
            index.add(i, make_vector(dim, i), i as f64).unwrap();
        }
        index.build().unwrap();

        // Unfiltered search should return results
        let query = make_vector(dim, 0);
        let results = index.search(&query, 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn narrow_range_returns_subset() {
        let dim = 16;
        let mut index = EsgIndex::new(
            dim,
            EsgParams {
                num_checkpoints: 4,
                hnsw_m: 8,
                hnsw_ef_construction: 50,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..40u32 {
            index.add(i, make_vector(dim, i), i as f64).unwrap();
        }
        index.build().unwrap();

        // Very narrow range: only 3 vectors qualify
        let query = make_vector(dim, 10);
        let results = index.range_search(&query, 10, 9.0, 11.0).unwrap();

        // At most 3 results (doc_ids 9, 10, 11)
        assert!(results.len() <= 3);
        for (doc_id, _) in &results {
            assert!(
                *doc_id >= 9 && *doc_id <= 11,
                "unexpected doc_id {} in narrow range",
                doc_id
            );
        }
    }

    #[test]
    fn empty_range_returns_empty() {
        let dim = 16;
        let mut index = EsgIndex::new(
            dim,
            EsgParams {
                num_checkpoints: 4,
                hnsw_m: 8,
                hnsw_ef_construction: 50,
                ef_search: 50,
            },
        )
        .unwrap();

        for i in 0..20u32 {
            index.add(i, make_vector(dim, i), i as f64).unwrap();
        }
        index.build().unwrap();

        // Range with no vectors
        let query = make_vector(dim, 0);
        let results = index.range_search(&query, 5, 100.0, 200.0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn geometric_checkpoints_basic() {
        let cp = geometric_checkpoints(100, 5);
        assert_eq!(cp.len(), 5);
        assert_eq!(cp[0], 0);
        assert_eq!(*cp.last().unwrap(), 99);

        let cp = geometric_checkpoints(10, 3);
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0], 0);
        assert_eq!(*cp.last().unwrap(), 9);

        let cp = geometric_checkpoints(5, 10);
        assert_eq!(cp, vec![0, 1, 2, 3, 4]);
    }
}
