//! Python bindings for vicinity (PyO3 + NumPy).
//!
//! Exposes [`HNSWIndex`] as the primary Python-facing class, accepting and
//! returning NumPy arrays for zero-copy interop where possible.

use std::borrow::Cow;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distance::{self, DistanceMetric as RustMetric};
use crate::hnsw::{HNSWIndex as RustHNSW, HNSWParams};

/// Distance metric for vector comparison.
#[pyclass(name = "DistanceMetric", module = "pyvicinity", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyDistanceMetric {
    /// Euclidean (L2) distance.
    L2,
    /// Cosine distance: `1 - cos(a, b)`.
    Cosine,
    /// Angular distance: `arccos(cos(a, b)) / pi`, in `[0, 1]`.
    Angular,
    /// Inner-product distance: `-dot(a, b)` (for MIPS).
    InnerProduct,
}

impl From<PyDistanceMetric> for RustMetric {
    fn from(m: PyDistanceMetric) -> Self {
        match m {
            PyDistanceMetric::L2 => RustMetric::L2,
            PyDistanceMetric::Cosine => RustMetric::Cosine,
            PyDistanceMetric::Angular => RustMetric::Angular,
            PyDistanceMetric::InnerProduct => RustMetric::InnerProduct,
        }
    }
}

impl From<RustMetric> for PyDistanceMetric {
    fn from(m: RustMetric) -> Self {
        match m {
            RustMetric::L2 => PyDistanceMetric::L2,
            RustMetric::Cosine => PyDistanceMetric::Cosine,
            RustMetric::Angular => PyDistanceMetric::Angular,
            RustMetric::InnerProduct => PyDistanceMetric::InnerProduct,
        }
    }
}

/// HNSW index for approximate nearest-neighbor search.
///
/// Example::
///
///     import numpy as np
///     from pyvicinity import HNSWIndex, DistanceMetric
///
///     index = HNSWIndex(dim=128, metric=DistanceMetric.Cosine, auto_normalize=True)
///     vectors = np.random.randn(10000, 128).astype(np.float32)
///     index.add_items(vectors)
///     index.build()
///     ids, dists = index.search(vectors[0], k=10, ef=50)
#[pyclass(name = "HNSWIndex", module = "pyvicinity")]
pub struct PyHNSWIndex {
    inner: RustHNSW,
    ef_search: usize,
    auto_normalize: bool,
    metric: PyDistanceMetric,
    m: usize,
    ef_construction: usize,
}

#[pymethods]
impl PyHNSWIndex {
    /// Create a new HNSW index.
    ///
    /// Args:
    ///     dim: Vector dimension.
    ///     m: Max connections per node (default 16).
    ///     ef_construction: Search width during build (default 200).
    ///     ef_search: Default search width for queries (default 50).
    ///     metric: Distance metric (default Cosine).
    ///     auto_normalize: L2-normalize vectors on insert AND query
    ///         (default False). Applies on both sides so cosine search on
    ///         non-unit-norm inputs returns sensible distances.
    ///     seed: RNG seed for reproducible builds (default None).
    #[new]
    #[pyo3(signature = (dim, m=16, ef_construction=200, ef_search=50, metric=PyDistanceMetric::Cosine, auto_normalize=false, seed=None))]
    fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: PyDistanceMetric,
        auto_normalize: bool,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if auto_normalize && !matches!(metric, PyDistanceMetric::Cosine | PyDistanceMetric::Angular)
        {
            return Err(PyValueError::new_err(format!(
                "auto_normalize=True is only meaningful for Cosine and Angular metrics; \
                 got {metric:?}. Use Cosine if you want spherical / unit-norm semantics, \
                 or set auto_normalize=False."
            )));
        }

        let params = HNSWParams {
            m,
            m_max: m * 2,
            ef_construction,
            ef_search,
            auto_normalize,
            metric: metric.into(),
            seed,
            ..Default::default()
        };
        let inner =
            RustHNSW::with_params(dim, params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            ef_search,
            auto_normalize,
            metric,
            m,
            ef_construction,
        })
    }

    /// Add vectors with optional explicit IDs.
    ///
    /// Args:
    ///     vectors: 2-D float32 array of shape ``(n, dim)``.
    ///     ids: Optional 1-D uint32 array of IDs. If None, assigns
    ///         sequential IDs starting from the current ``len(index)``.
    #[pyo3(signature = (vectors, ids=None))]
    fn add_items<'py>(
        &mut self,
        vectors: PyReadonlyArray2<'py, f32>,
        ids: Option<PyReadonlyArray1<'py, u32>>,
    ) -> PyResult<()> {
        let arr = vectors.as_array();
        let (n, d) = (arr.nrows(), arr.ncols());

        if d != self.inner.dimension {
            return Err(PyValueError::new_err(format!(
                "dimension mismatch: index expects {}, got {d}",
                self.inner.dimension
            )));
        }

        let data = vectors
            .as_slice()
            .map_err(|_| PyValueError::new_err("vectors must be contiguous (C-order)"))?;

        match ids {
            Some(id_arr) => {
                let id_slice = id_arr
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("ids must be contiguous"))?;
                if id_slice.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "ids length {} != vectors rows {n}",
                        id_slice.len()
                    )));
                }
                self.inner
                    .add_batch(id_slice, data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            None => {
                let base = self.inner.num_vectors as u32;
                let id_vec: Vec<u32> = (base..base + n as u32).collect();
                self.inner
                    .add_batch(&id_vec, data)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        Ok(())
    }

    /// Finalize the index. Must be called after all vectors are added and
    /// before any search.
    fn build(&mut self) -> PyResult<()> {
        self.inner
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Set the default `ef_search` parameter for subsequent queries.
    fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    /// Search for k nearest neighbors of a single query vector.
    ///
    /// Args:
    ///     query: 1-D float32 array of shape ``(dim,)``.
    ///     k: Number of neighbors to return.
    ///     ef: Search width (overrides default ef_search if provided).
    ///
    /// Returns:
    ///     Tuple ``(ids, distances)`` of 1-D arrays. Length is at most k;
    ///     fewer if the index has fewer than k vectors.
    #[pyo3(signature = (query, k, ef=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
        let q = query
            .as_slice()
            .map_err(|_| PyValueError::new_err("query must be contiguous"))?;
        if q.len() != self.inner.dimension {
            return Err(PyValueError::new_err(format!(
                "query dimension mismatch: index expects {}, got {}",
                self.inner.dimension,
                q.len()
            )));
        }

        let ef = ef.unwrap_or(self.ef_search);
        let prepared = prep_query(q, self.metric, self.auto_normalize);

        let results = py
            .detach(|| self.inner.search(prepared.as_ref(), k, ef))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let n = results.len();
        let mut ids = Vec::with_capacity(n);
        let mut dists = Vec::with_capacity(n);
        for (id, dist) in &results {
            ids.push(*id);
            dists.push(*dist);
        }
        Ok((ids.into_pyarray(py), dists.into_pyarray(py)))
    }

    /// Batch search: find k nearest neighbors for each query.
    ///
    /// Args:
    ///     queries: 2-D float32 array of shape ``(nq, dim)``.
    ///     k: Number of neighbors per query.
    ///     ef: Search width (overrides default ef_search if provided).
    ///
    /// Returns:
    ///     Tuple ``(ids, distances)`` of 2-D arrays of shape ``(nq, k)``.
    ///     Rows with fewer than k results are padded with ``u32::MAX`` /
    ///     ``inf`` so the array is rectangular.
    #[pyo3(signature = (queries, k, ef=None))]
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let dim = arr.ncols();
        let ef = ef.unwrap_or(self.ef_search);

        if dim != self.inner.dimension {
            return Err(PyValueError::new_err(format!(
                "queries dimension mismatch: index expects {}, got {dim}",
                self.inner.dimension
            )));
        }

        let data = queries
            .as_slice()
            .map_err(|_| PyValueError::new_err("queries must be contiguous (C-order)"))?;

        let mut all_ids = vec![u32::MAX; nq * k];
        let mut all_dists = vec![f32::INFINITY; nq * k];

        let metric = self.metric;
        let auto_normalize = self.auto_normalize;

        py.detach(|| {
            for i in 0..nq {
                let q = &data[i * dim..(i + 1) * dim];
                let prepared = prep_query(q, metric, auto_normalize);
                if let Ok(results) = self.inner.search(prepared.as_ref(), k, ef) {
                    for (j, (id, dist)) in results.iter().enumerate() {
                        all_ids[i * k + j] = *id;
                        all_dists[i * k + j] = *dist;
                    }
                }
            }
        });

        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, k), all_ids)
            .map_err(|e| PyValueError::new_err(format!("failed to reshape ids: {e}")))?;
        let dists_arr = numpy::ndarray::Array2::from_shape_vec((nq, k), all_dists)
            .map_err(|e| PyValueError::new_err(format!("failed to reshape dists: {e}")))?;

        Ok((ids_arr.into_pyarray(py), dists_arr.into_pyarray(py)))
    }

    /// Number of vectors in the index.
    #[getter]
    fn num_vectors(&self) -> usize {
        self.inner.num_vectors
    }

    /// Vector dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension
    }

    /// Distance metric this index was built with.
    #[getter]
    fn metric(&self) -> PyDistanceMetric {
        self.metric
    }

    /// Whether this index normalizes inserts and queries.
    #[getter]
    fn auto_normalize(&self) -> bool {
        self.auto_normalize
    }

    /// Max connections per node (the `M` parameter).
    #[getter]
    fn m(&self) -> usize {
        self.m
    }

    /// Search width during construction.
    #[getter]
    fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Default search width for queries.
    #[getter]
    fn ef_search(&self) -> usize {
        self.ef_search
    }

    /// Number of vectors in the index. Enables ``len(index)``.
    fn __len__(&self) -> usize {
        self.inner.num_vectors
    }

    fn __repr__(&self) -> String {
        let metric = match self.metric {
            PyDistanceMetric::L2 => "L2",
            PyDistanceMetric::Cosine => "Cosine",
            PyDistanceMetric::Angular => "Angular",
            PyDistanceMetric::InnerProduct => "InnerProduct",
        };
        format!(
            "HNSWIndex(dim={}, n={}, metric=DistanceMetric.{}, m={}, ef_construction={}, ef_search={}, auto_normalize={})",
            self.inner.dimension,
            self.inner.num_vectors,
            metric,
            self.m,
            self.ef_construction,
            self.ef_search,
            if self.auto_normalize { "True" } else { "False" },
        )
    }
}

/// Normalize the query if the metric needs it and `auto_normalize` is on.
///
/// Cosine uses the dot-only fast path internally (`cosine_distance_normalized`),
/// so a non-unit query produces meaningless distances. Angular and L2 are
/// scale-aware on the query side already; InnerProduct is intentionally not
/// normalized (MIPS semantics).
fn prep_query<'a>(
    query: &'a [f32],
    metric: PyDistanceMetric,
    auto_normalize: bool,
) -> Cow<'a, [f32]> {
    if auto_normalize && matches!(metric, PyDistanceMetric::Cosine) {
        Cow::Owned(distance::normalize(query))
    } else {
        Cow::Borrowed(query)
    }
}

/// Register the Python module.
///
/// The module name (`_core`) must match the last path segment of
/// `module-name` in `pyproject.toml` (`pyvicinity._core`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Sentinel values used to pad short rows in `batch_search` results.
    // Exported so callers have a stable name to mask against, e.g.
    // `labels[labels != pyvicinity.MISSING_LABEL]`.
    m.add("MISSING_LABEL", u32::MAX)?;
    m.add("MISSING_DISTANCE", f32::INFINITY)?;
    m.add_class::<PyDistanceMetric>()?;
    m.add_class::<PyHNSWIndex>()?;
    Ok(())
}
