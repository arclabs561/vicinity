//! Python bindings for vicinity (PyO3 + NumPy).
//!
//! Exposes [`HNSWIndex`] as the primary Python-facing class, accepting and
//! returning NumPy arrays for zero-copy interop where possible.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distance::DistanceMetric as RustMetric;
use crate::hnsw::{HNSWIndex as RustHNSW, HNSWParams};

/// Distance metric for vector comparison.
#[pyclass(name = "DistanceMetric", eq)]
#[derive(Clone, Copy, PartialEq)]
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

/// HNSW index for approximate nearest-neighbor search.
///
/// Example::
///
///     import numpy as np
///     from vicinity import HNSWIndex, DistanceMetric
///
///     index = HNSWIndex(dim=128, metric=DistanceMetric.Cosine)
///     vectors = np.random.randn(10000, 128).astype(np.float32)
///     index.add_items(vectors)
///     index.build()
///     ids, dists = index.search(vectors[0], k=10, ef=50)
#[pyclass(name = "HNSWIndex")]
pub struct PyHNSWIndex {
    inner: RustHNSW,
    ef_search: usize,
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
    ///     auto_normalize: L2-normalize vectors on insert (default False).
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
        Ok(Self { inner, ef_search })
    }

    /// Add vectors with auto-assigned sequential IDs.
    ///
    /// Args:
    ///     vectors: 2-D float32 array of shape ``(n, dim)``.
    ///     ids: Optional 1-D uint32 array of IDs. If None, assigns 0..n.
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

        // Get contiguous slice (numpy row-major = what we need).
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

    /// Finalize the index (must be called after all vectors are added).
    fn build(&mut self) -> PyResult<()> {
        self.inner
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Set the default ef_search parameter for subsequent queries.
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
    ///     Tuple of ``(ids, distances)`` — both 1-D arrays of length k.
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

        let ef = ef.unwrap_or(self.ef_search);
        let results = self
            .inner
            .search(q, k, ef)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut ids = Vec::with_capacity(results.len());
        let mut dists = Vec::with_capacity(results.len());
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
    ///     Tuple of ``(ids, distances)`` — both 2-D arrays of shape ``(nq, k)``.
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
        let ef = ef.unwrap_or(self.ef_search);

        let data = queries
            .as_slice()
            .map_err(|_| PyValueError::new_err("queries must be contiguous (C-order)"))?;

        let dim = self.inner.dimension;
        let mut all_ids = Vec::with_capacity(nq * k);
        let mut all_dists = Vec::with_capacity(nq * k);

        // Release GIL during the search-intensive loop.
        py.detach(|| {
            for i in 0..nq {
                let q = &data[i * dim..(i + 1) * dim];
                match self.inner.search(q, k, ef) {
                    Ok(results) => {
                        for (id, dist) in &results {
                            all_ids.push(*id);
                            all_dists.push(*dist);
                        }
                        // Pad if fewer results than k.
                        for _ in results.len()..k {
                            all_ids.push(u32::MAX);
                            all_dists.push(f32::INFINITY);
                        }
                    }
                    Err(_) => {
                        for _ in 0..k {
                            all_ids.push(u32::MAX);
                            all_dists.push(f32::INFINITY);
                        }
                    }
                }
            }
        });

        let ids_arr = numpy::ndarray::Array2::from_shape_vec((nq, k), all_ids)
            .map_err(|e| PyValueError::new_err(format!("failed to reshape ids: {e}")))?;
        let dists_arr = numpy::ndarray::Array2::from_shape_vec((nq, k), all_dists)
            .map_err(|e| PyValueError::new_err(format!("failed to reshape dists: {e}")))?;

        Ok((
            ids_arr.into_pyarray(py),
            dists_arr.into_pyarray(py),
        ))
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

    fn __repr__(&self) -> String {
        format!(
            "HNSWIndex(dim={}, n={}, ef_search={})",
            self.inner.dimension, self.inner.num_vectors, self.ef_search
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Register the Python module.
#[pymodule]
fn _vicinity(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMetric>()?;
    m.add_class::<PyHNSWIndex>()?;
    Ok(())
}
