//! Range-filtered ANN search (HNSW + attribute-range post-filter).
//!
//! Handles queries of the form "find k nearest neighbors of q among vectors
//! whose attribute falls in `[l, r]`". Builds a single HNSW index over all
//! vectors and post-filters search results by the attribute range.
//!
//! Renamed from `esg` in 0.8.0. The original name implied fidelity to Yang
//! et al. 2025 (arXiv:2504.04018), which describes a partition-aware
//! structure (subrange-partitioned subgraphs with elastic relaxation,
//! reducing required subranges to at most two). The shipped implementation
//! is the paper's strawman baseline; the rename reflects what it does.
//! A paper-fidelity ESG variant is planned as a separate module.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.8", features = ["range_filtered"] } # 0.8.0+
//! # or, on 0.7.x: features = ["esg"]
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::range_filtered::{RangeFilteredIndex, RangeFilteredParams};
//!
//! let params = RangeFilteredParams::default();
//! let mut index = RangeFilteredIndex::new(128, params)?;
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
//! 1. Sort all vectors by their numeric attribute.
//! 2. Build a single HNSW index over all vectors.
//! 3. For range queries, search the full index with ef * 2 candidates and
//!    post-filter to retain only results whose attribute falls in [lo, hi].
//!
//! # References
//!
//! - Yang et al. (2025). "ESG: Elastic Graphs for Range-Filtering Approximate
//!   kNN Search." arXiv:2504.04018. Cited for context only -- this module
//!   does *not* implement the paper's partition-aware structure; it ships
//!   the post-filter baseline.

use crate::RetrieveError;

/// Range-filtered index parameters.
///
/// The shipped implementation is HNSW + attribute-range post-filter.
/// For an explicit predicate filter on top of HNSW, see
/// [`crate::hnsw::filtered`] / ACORN.
#[derive(Clone, Debug)]
pub struct RangeFilteredParams {
    /// HNSW M parameter for index construction.
    pub hnsw_m: usize,
    /// HNSW ef_construction.
    pub hnsw_ef_construction: usize,
    /// Search ef parameter.
    pub ef_search: usize,
}

impl Default for RangeFilteredParams {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            ef_search: 100,
        }
    }
}

/// A point with a vector and a numeric attribute.
#[derive(Clone, Debug)]
struct AttributedPoint {
    doc_id: u32,
    attribute: f64,
}

/// Range-filtered ANN index (HNSW + attribute-range post-filter).
pub struct RangeFilteredIndex {
    dimension: usize,
    params: RangeFilteredParams,
    built: bool,

    /// Normalized vectors in sorted-by-attribute order.
    vectors: Vec<f32>,
    num_vectors: usize,
    /// Points sorted by attribute (populated during build).
    sorted_points: Vec<AttributedPoint>,
    /// Unsorted points (pre-build staging).
    staging: Vec<(u32, Vec<f32>, f64)>,

    /// Single HNSW index over all vectors.
    #[cfg(feature = "hnsw")]
    full_index: Option<crate::hnsw::HNSWIndex>,
}

impl RangeFilteredIndex {
    /// Create a new range-filtered index.
    pub fn new(dimension: usize, params: RangeFilteredParams) -> Result<Self, RetrieveError> {
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
            full_index: None,
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

    /// Build the index: sort by attribute, build a single HNSW over all vectors.
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

        for (doc_id, vector, attribute) in self.staging.drain(..) {
            self.sorted_points
                .push(AttributedPoint { doc_id, attribute });
            // L2-normalize
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                self.vectors.extend(vector.iter().map(|x| x / norm));
            } else {
                self.vectors.extend_from_slice(&vector);
            }
        }

        // Build a single HNSW index over all vectors
        let mut hnsw = crate::hnsw::HNSWIndex::builder(self.dimension)
            .m(self.params.hnsw_m)
            .ef_construction(self.params.hnsw_ef_construction)
            .auto_normalize(false)
            .build()?;

        for (rank, point) in self.sorted_points.iter().enumerate() {
            let vec = self.get_vector(rank);
            hnsw.add_slice(point.doc_id, vec)?;
        }
        hnsw.build()?;

        self.full_index = Some(hnsw);
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

        let hnsw = match self.full_index.as_ref() {
            Some(h) => h,
            None => {
                return Err(RetrieveError::InvalidParameter(
                    "index must be built before search".into(),
                ))
            }
        };

        // Normalize query
        let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let query_normalized: Vec<f32> = if query_norm > 1e-10 {
            query.iter().map(|x| x / query_norm).collect()
        } else {
            query.to_vec()
        };

        let ef = self.params.ef_search.max(k);

        // Build attribute lookup: doc_id -> attribute
        // sorted_points is small; linear scan per result is acceptable.
        let in_range = |doc_id: u32| -> bool {
            self.sorted_points
                .iter()
                .any(|p| p.doc_id == doc_id && p.attribute >= lo && p.attribute <= hi)
        };

        // Search with extra candidates, then post-filter by attribute range
        let candidates = hnsw.search(&query_normalized, k * 4, ef * 2)?;
        let mut results: Vec<(u32, f32)> = candidates
            .into_iter()
            .filter(|(doc_id, _)| in_range(*doc_id))
            .take(k)
            .collect();

        results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);

        Ok(results)
    }

    /// Unfiltered search (searches the full index).
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

#[cfg(test)]
#[cfg(feature = "hnsw")]
#[allow(clippy::unwrap_used, deprecated)]
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
        let mut index = RangeFilteredIndex::new(
            dim,
            RangeFilteredParams {
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
                (20.0..=60.0).contains(&attr),
                "doc_id {} has attribute {}, expected in [20, 60]",
                doc_id,
                attr
            );
        }
    }

    #[test]
    fn full_range_search() {
        let dim = 16;
        let mut index = RangeFilteredIndex::new(
            dim,
            RangeFilteredParams {
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
        let mut index = RangeFilteredIndex::new(
            dim,
            RangeFilteredParams {
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
        let mut index = RangeFilteredIndex::new(
            dim,
            RangeFilteredParams {
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
}
