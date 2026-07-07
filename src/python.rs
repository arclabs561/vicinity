//! Python bindings for vicinity (PyO3 + NumPy).
//!
//! Exposes [`PyHNSWIndex`] (renamed `HNSWIndex` on the Python side) as the
//! primary Python-facing class. Inputs are accepted as zero-copy NumPy
//! views where possible (`PyReadonlyArray`); outputs are owned arrays
//! produced via `into_pyarray`.

use std::borrow::Cow;
use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distance::{self, DistanceMetric as RustMetric};
use crate::hnsw::{HNSWIndex as RustHNSW, HNSWParams};
use crate::ivf_pq::{IVFPQIndex as RustIVFPQ, IVFPQParams};

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
/// Example:
///
/// ```python
/// import numpy as np
/// from pyvicinity import HNSWIndex, DistanceMetric
///
/// index = HNSWIndex(dim=128, metric=DistanceMetric.Cosine, auto_normalize=True)
/// vectors = np.random.randn(10000, 128).astype(np.float32)
/// index.add_items(vectors)
/// index.build()
/// ids, dists = index.search(vectors[0], k=10, ef=50)
/// ```
#[pyclass(name = "HNSWIndex", module = "pyvicinity")]
pub struct PyHNSWIndex {
    inner: RustHNSW,
    ef_search: usize,
    auto_normalize: bool,
    metric: PyDistanceMetric,
    m: usize,
    ef_construction: usize,
}

/// IVF-PQ index for compressed approximate nearest-neighbor search.
///
/// Vectors and queries are L2-normalized internally; this index is intended
/// for cosine-style dense vector search. Use `rerank_pool` in `search` or
/// `batch_search` when exact f32 reranking is needed.
#[pyclass(name = "IVFPQIndex", module = "pyvicinity")]
pub struct PyIVFPQIndex {
    inner: RustIVFPQ,
    nprobe: usize,
    num_clusters: usize,
    num_codebooks: usize,
    codebook_size: usize,
    use_opq: bool,
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
    ///     ids: Optional 1-D int64 array of IDs. Each value must be in
    ///         ``[0, 2**32)``. If None, sequential IDs are assigned
    ///         starting from the current ``len(index)``.
    #[pyo3(signature = (vectors, ids=None))]
    fn add_items<'py>(
        &mut self,
        vectors: PyReadonlyArray2<'py, f32>,
        ids: Option<PyReadonlyArray1<'py, i64>>,
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
                let mut id_u32 = Vec::with_capacity(id_slice.len());
                for (i, &id) in id_slice.iter().enumerate() {
                    if !(0..=u32::MAX as i64).contains(&id) {
                        return Err(PyValueError::new_err(format!(
                            "ids[{i}] = {id} out of range [0, 2**32); pyvicinity stores IDs as u32 internally",
                        )));
                    }
                    id_u32.push(id as u32);
                }
                self.inner
                    .add_batch(&id_u32, data)
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
        self.inner.params.ef_search = ef;
    }

    /// Save this index to a JSON snapshot file.
    ///
    /// The underlying Rust snapshot does not persist tombstones or metadata.
    /// pyvicinity does not expose deletion or filtered-search metadata today,
    /// so this round-trip preserves the current Python API surface.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner
            .save_to_file(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load an index from a JSON snapshot file written by `save`.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner =
            RustHNSW::load_from_file(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            ef_search: inner.params.ef_search,
            auto_normalize: inner.params.auto_normalize,
            metric: inner.params.metric.into(),
            m: inner.params.m,
            ef_construction: inner.params.ef_construction,
            inner,
        })
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
    ///     fewer if the index has fewer than k vectors. ``ids`` are int64
    ///     (faiss-compatible).
    #[pyo3(signature = (query, k, ef=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
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
            ids.push(*id as i64);
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
    ///     Rows with fewer than k results are padded with ``-1`` (int64)
    ///     and ``+inf`` so the array is rectangular. The sentinel matches
    ///     faiss's convention; mask with ``ids != -1`` (or the named
    ///     ``pyvicinity.MISSING_LABEL``).
    #[pyo3(signature = (queries, k, ef=None))]
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
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

        let mut all_ids = vec![-1i64; nq * k];
        let mut all_dists = vec![f32::INFINITY; nq * k];

        let metric = self.metric;
        let auto_normalize = self.auto_normalize;

        let batch_results = py
            .detach(|| -> Result<Vec<Vec<(u32, f32)>>, crate::RetrieveError> {
                #[cfg(feature = "parallel")]
                {
                    let prepared = prep_batch_queries(data, nq, dim, metric, auto_normalize);
                    self.inner.search_batch_flat(prepared.as_ref(), nq, k, ef)
                }
                #[cfg(not(feature = "parallel"))]
                {
                    let mut results = Vec::with_capacity(nq);
                    for i in 0..nq {
                        let q = &data[i * dim..(i + 1) * dim];
                        let prepared = prep_query(q, metric, auto_normalize);
                        results.push(self.inner.search(prepared.as_ref(), k, ef)?);
                    }
                    Ok(results)
                }
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        for (i, results) in batch_results.iter().enumerate() {
            for (j, (id, dist)) in results.iter().enumerate() {
                all_ids[i * k + j] = *id as i64;
                all_dists[i * k + j] = *dist;
            }
        }

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

    /// Estimated memory used by this index, in bytes.
    #[getter]
    fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage().total()
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

#[pymethods]
impl PyIVFPQIndex {
    /// Create a new IVF-PQ index.
    ///
    /// Args:
    ///     dim: Vector dimension.
    ///     num_clusters: Number of IVF coarse clusters.
    ///     num_codebooks: Number of PQ codebooks. Defaults to the largest
    ///         divisor of ``dim`` up to 8.
    ///     codebook_size: Number of centroids per PQ codebook, in ``1..=256``.
    ///     nprobe: Number of coarse clusters to scan per query.
    ///     use_opq: Enable OPQ rotation during build.
    ///     seed: RNG seed for reproducible training.
    #[new]
    #[pyo3(signature = (dim, num_clusters=256, num_codebooks=None, codebook_size=256, nprobe=1, use_opq=false, seed=42))]
    fn new(
        dim: usize,
        num_clusters: usize,
        num_codebooks: Option<usize>,
        codebook_size: usize,
        nprobe: usize,
        use_opq: bool,
        seed: u64,
    ) -> PyResult<Self> {
        let num_codebooks = num_codebooks.unwrap_or_else(|| default_ivfpq_codebooks(dim));
        validate_ivfpq_shape(dim, num_clusters, num_codebooks, codebook_size, nprobe)?;
        let params = IVFPQParams {
            num_clusters,
            nprobe,
            num_codebooks,
            codebook_size,
            use_opq,
            seed,
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: IVFPQParams::default().compression_threshold,
        };
        let inner =
            RustIVFPQ::new(dim, params).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            nprobe,
            num_clusters,
            num_codebooks,
            codebook_size,
            use_opq,
        })
    }

    /// Add vectors with optional explicit IDs.
    ///
    /// Args:
    ///     vectors: 2-D float32 array of shape ``(n, dim)``.
    ///     ids: Optional 1-D int64 array of IDs. Each value must be in
    ///         ``[0, 2**32)``. If None, sequential IDs are assigned
    ///         starting from the current ``len(index)``.
    #[pyo3(signature = (vectors, ids=None))]
    fn add_items<'py>(
        &mut self,
        vectors: PyReadonlyArray2<'py, f32>,
        ids: Option<PyReadonlyArray1<'py, i64>>,
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

        let id_u32 = match ids {
            Some(id_arr) => {
                let id_slice = id_arr
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("ids must be contiguous"))?;
                checked_ids(id_slice, n)?
            }
            None => {
                let base = self.inner.num_vectors as u32;
                (base..base + n as u32).collect()
            }
        };

        for (row, &doc_id) in data.chunks_exact(d).zip(id_u32.iter()) {
            self.inner
                .add_slice(doc_id, row)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Finalize the index.
    ///
    /// Args:
    ///     training_sample_size: Optional deterministic sample size for
    ///         IVF/PQ k-means training. All added vectors remain searchable.
    ///     kmeans_max_iter: Maximum Lloyd iterations for IVF and PQ training.
    #[pyo3(signature = (training_sample_size=None, kmeans_max_iter=100))]
    fn build(
        &mut self,
        training_sample_size: Option<usize>,
        kmeans_max_iter: usize,
    ) -> PyResult<()> {
        self.inner
            .build_with_training_options(training_sample_size, kmeans_max_iter)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Set the default ``nprobe`` parameter for subsequent queries.
    fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe;
        self.inner.set_nprobe(nprobe);
    }

    /// Drop raw f32 vectors after build. Approximate search still works, but
    /// exact reranking is no longer available.
    fn compact(&mut self) -> PyResult<()> {
        if !self.inner.is_built() {
            return Err(PyValueError::new_err("index must be built before compact"));
        }
        self.inner.compact();
        Ok(())
    }

    /// Save this index to a directory snapshot.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner
            .save_to_dir(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load an index from a directory snapshot written by `save`.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        let inner =
            RustIVFPQ::load_from_dir(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            nprobe: inner.nprobe(),
            num_clusters: inner.num_clusters(),
            num_codebooks: inner.num_codebooks(),
            codebook_size: inner.codebook_size(),
            use_opq: inner.use_opq(),
            inner,
        })
    }

    /// Search for k nearest neighbors of a single query vector.
    ///
    /// Args:
    ///     query: 1-D float32 array of shape ``(dim,)``.
    ///     k: Number of neighbors to return.
    ///     nprobe: Number of IVF clusters to scan for this call. Defaults to
    ///         ``self.nprobe`` and does not change the default.
    ///     rerank_pool: If provided, rerank this many approximate candidates
    ///         using exact f32 cosine distance.
    #[pyo3(signature = (query, k, nprobe=None, rerank_pool=None))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        query: PyReadonlyArray1<'py, f32>,
        k: usize,
        nprobe: Option<usize>,
        rerank_pool: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
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

        let results = self.search_one(py, q, k, nprobe, rerank_pool)?;
        let n = results.len();
        let mut ids = Vec::with_capacity(n);
        let mut dists = Vec::with_capacity(n);
        for (id, dist) in &results {
            ids.push(*id as i64);
            dists.push(*dist);
        }
        Ok((ids.into_pyarray(py), dists.into_pyarray(py)))
    }

    /// Batch search: find k nearest neighbors for each query.
    ///
    /// Rows with fewer than k results are padded with ``MISSING_LABEL`` and
    /// ``MISSING_DISTANCE``.
    #[pyo3(signature = (queries, k, nprobe=None, rerank_pool=None))]
    fn batch_search<'py>(
        &mut self,
        py: Python<'py>,
        queries: PyReadonlyArray2<'py, f32>,
        k: usize,
        nprobe: Option<usize>,
        rerank_pool: Option<usize>,
    ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
        let arr = queries.as_array();
        let nq = arr.nrows();
        let dim = arr.ncols();
        if dim != self.inner.dimension {
            return Err(PyValueError::new_err(format!(
                "queries dimension mismatch: index expects {}, got {dim}",
                self.inner.dimension
            )));
        }

        let data = queries
            .as_slice()
            .map_err(|_| PyValueError::new_err("queries must be contiguous (C-order)"))?;

        let batch_results = py
            .detach(|| -> Result<Vec<Vec<(u32, f32)>>, crate::RetrieveError> {
                let mut results = Vec::with_capacity(nq);
                for i in 0..nq {
                    let q = &data[i * dim..(i + 1) * dim];
                    results.push(self.search_one_inner(q, k, nprobe, rerank_pool)?);
                }
                Ok(results)
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut all_ids = vec![-1i64; nq * k];
        let mut all_dists = vec![f32::INFINITY; nq * k];
        for (i, results) in batch_results.iter().enumerate() {
            for (j, (id, dist)) in results.iter().enumerate() {
                all_ids[i * k + j] = *id as i64;
                all_dists[i * k + j] = *dist;
            }
        }

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

    /// Number of IVF coarse clusters.
    #[getter]
    fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Number of PQ codebooks.
    #[getter]
    fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Number of centroids per PQ codebook.
    #[getter]
    fn codebook_size(&self) -> usize {
        self.codebook_size
    }

    /// Default IVF clusters scanned per query.
    #[getter]
    fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Whether OPQ rotation is enabled.
    #[getter]
    fn use_opq(&self) -> bool {
        self.use_opq
    }

    /// Number of vectors in the index. Enables ``len(index)``.
    fn __len__(&self) -> usize {
        self.inner.num_vectors
    }

    fn __repr__(&self) -> String {
        format!(
            "IVFPQIndex(dim={}, n={}, num_clusters={}, num_codebooks={}, codebook_size={}, nprobe={}, use_opq={})",
            self.inner.dimension,
            self.inner.num_vectors,
            self.num_clusters,
            self.num_codebooks,
            self.codebook_size,
            self.nprobe,
            if self.use_opq { "True" } else { "False" },
        )
    }
}

impl PyIVFPQIndex {
    fn search_one(
        &mut self,
        py: Python<'_>,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
        rerank_pool: Option<usize>,
    ) -> PyResult<Vec<(u32, f32)>> {
        py.detach(|| self.search_one_inner(query, k, nprobe, rerank_pool))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn search_one_inner(
        &mut self,
        query: &[f32],
        k: usize,
        nprobe: Option<usize>,
        rerank_pool: Option<usize>,
    ) -> Result<Vec<(u32, f32)>, crate::RetrieveError> {
        let saved_nprobe = self.nprobe;
        if let Some(nprobe) = nprobe {
            self.inner.set_nprobe(nprobe);
        }

        let result = match rerank_pool {
            Some(pool) => self.inner.search_reranked(query, k, pool),
            None => self.inner.search(query, k),
        };

        if nprobe.is_some() {
            self.inner.set_nprobe(saved_nprobe);
        }

        result
    }
}

fn default_ivfpq_codebooks(dim: usize) -> usize {
    (1..=8.min(dim))
        .rev()
        .find(|&codebooks| dim.is_multiple_of(codebooks))
        .unwrap_or(1)
}

fn validate_ivfpq_shape(
    dim: usize,
    num_clusters: usize,
    num_codebooks: usize,
    codebook_size: usize,
    nprobe: usize,
) -> PyResult<()> {
    if dim == 0 {
        return Err(PyValueError::new_err("dim must be greater than 0"));
    }
    if num_clusters == 0 {
        return Err(PyValueError::new_err("num_clusters must be greater than 0"));
    }
    if num_codebooks == 0 || !dim.is_multiple_of(num_codebooks) {
        return Err(PyValueError::new_err(format!(
            "num_codebooks must be a non-zero divisor of dim (dim={dim}, num_codebooks={num_codebooks})"
        )));
    }
    if codebook_size == 0 || codebook_size > 256 {
        return Err(PyValueError::new_err(
            "codebook_size must be in the range 1..=256",
        ));
    }
    if nprobe == 0 {
        return Err(PyValueError::new_err("nprobe must be greater than 0"));
    }
    Ok(())
}

fn checked_ids(ids: &[i64], expected_len: usize) -> PyResult<Vec<u32>> {
    if ids.len() != expected_len {
        return Err(PyValueError::new_err(format!(
            "ids length {} != vectors rows {expected_len}",
            ids.len()
        )));
    }
    let mut id_u32 = Vec::with_capacity(ids.len());
    for (i, &id) in ids.iter().enumerate() {
        if !(0..=u32::MAX as i64).contains(&id) {
            return Err(PyValueError::new_err(format!(
                "ids[{i}] = {id} out of range [0, 2**32); pyvicinity stores IDs as u32 internally",
            )));
        }
        id_u32.push(id as u32);
    }
    Ok(id_u32)
}

/// Normalize the query if `auto_normalize` is on and the metric supports it.
///
/// Cosine *requires* query normalization: the index uses the dot-only fast
/// path (`cosine_distance_normalized`), so a non-unit query produces
/// meaningless distances. Angular doesn't strictly require it (the underlying
/// `angular_distance` re-computes norms), but we normalize it too for
/// symmetric behavior with the `auto_normalize` flag name -- the cost is one
/// allocation per query, and the alternative (silent asymmetry) ages badly if
/// the underlying distance function ever changes. L2 and InnerProduct never
/// reach this branch because the constructor rejects `auto_normalize=True`
/// for those metrics.
fn prep_query<'a>(
    query: &'a [f32],
    metric: PyDistanceMetric,
    auto_normalize: bool,
) -> Cow<'a, [f32]> {
    if auto_normalize && matches!(metric, PyDistanceMetric::Cosine | PyDistanceMetric::Angular) {
        Cow::Owned(distance::normalize(query))
    } else {
        Cow::Borrowed(query)
    }
}

#[cfg(feature = "parallel")]
fn prep_batch_queries<'a>(
    queries: &'a [f32],
    num_queries: usize,
    dim: usize,
    metric: PyDistanceMetric,
    auto_normalize: bool,
) -> Cow<'a, [f32]> {
    if auto_normalize && matches!(metric, PyDistanceMetric::Cosine | PyDistanceMetric::Angular) {
        let mut normalized = Vec::with_capacity(queries.len());
        for i in 0..num_queries {
            normalized.extend(distance::normalize(&queries[i * dim..(i + 1) * dim]));
        }
        Cow::Owned(normalized)
    } else {
        Cow::Borrowed(queries)
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
    // `labels[labels != pyvicinity.MISSING_LABEL]`. Matches faiss's
    // (-1, +inf) convention.
    m.add("MISSING_LABEL", -1i64)?;
    m.add("MISSING_DISTANCE", f32::INFINITY)?;
    m.add_class::<PyDistanceMetric>()?;
    m.add_class::<PyHNSWIndex>()?;
    m.add_class::<PyIVFPQIndex>()?;
    Ok(())
}
