"""ann-benchmarks / big-ann-benchmarks ``BaseANN`` wrapper for pyvicinity.

Conforms to the BaseANN interface expected by:

* ann-benchmarks (erikbern/ann-benchmarks)
* big-ann-benchmarks (harsha-simhadri/big-ann-benchmarks)
* VIBE (vector-index-bench/vibe)

Drop-in usage with ann-benchmarks: copy or symlink this module to
``ann_benchmarks/algorithms/pyvicinity/module.py`` and add the
matching ``config.yml`` and ``Dockerfile`` for the harness.

Standalone use::

    import numpy as np
    from pyvicinity.ann_benchmarks import VicinityHNSW, VicinityIVFPQ

    rng = np.random.default_rng(0)
    train = rng.standard_normal((10_000, 64), dtype=np.float32)

    algo = VicinityHNSW("cosine", {"M": 16, "efConstruction": 200})
    algo.fit(train)
    algo.set_query_arguments(100)            # ef_search
    ids = algo.query(train[0], 10)           # single-query, ann-benchmarks
    algo.batch_query(train[:32], 10)         # batch, big-ann-benchmarks/VIBE
    batch_ids = algo.get_batch_results()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from pyvicinity._core import DistanceMetric, HNSWIndex, IVFPQIndex


class VicinityHNSW:
    """ann-benchmarks-compatible HNSW wrapper around :class:`HNSWIndex`."""

    def __init__(self, metric: str, method_param: Mapping[str, Any]) -> None:
        self._metric_name = metric
        self._metric = _parse_metric(metric)
        self._m = int(cast(int, method_param.get("M", 16)))
        self._ef_construction = int(cast(int, method_param.get("efConstruction", 200)))
        self._ef_search = 50
        self._index: HNSWIndex | None = None
        self._batch_results: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Build the index from training data."""
        if X.ndim != 2:
            msg = f"fit() expects a 2-D array, got shape {X.shape}"
            raise ValueError(msg)
        _n, dim = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)

        # Angular/cosine harnesses commonly pass un-normalized inputs.
        auto_norm = self._metric_name.lower() in ("angular", "cosine")

        self._index = HNSWIndex(
            dim=dim,
            m=self._m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            metric=self._metric,
            auto_normalize=auto_norm,
        )
        self._index.add_items(X)
        self._index.build()

    def set_query_arguments(self, ef_search: int) -> None:
        """Set ef_search for subsequent queries."""
        self._ef_search = int(ef_search)
        if self._index is not None:
            self._index.set_ef_search(self._ef_search)

    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """Single-query interface (ann-benchmarks). Returns ids only."""
        index = self._require_index()
        q = np.ascontiguousarray(q, dtype=np.float32)
        ids, _dists = index.search(q, k=n, ef=self._ef_search)
        return ids

    def batch_query(self, X: np.ndarray, n: int) -> None:
        """Batch-query interface (big-ann-benchmarks / VIBE).

        Stores results internally; retrieve with :meth:`get_batch_results`.
        """
        index = self._require_index()
        X = np.ascontiguousarray(X, dtype=np.float32)
        ids, _dists = index.batch_search(X, k=n, ef=self._ef_search)
        self._batch_results = ids

    def get_batch_results(self) -> np.ndarray:
        """Return ids from the last :meth:`batch_query` call."""
        if self._batch_results is None:
            msg = "no batch results: call batch_query(...) first"
            raise RuntimeError(msg)
        return self._batch_results

    def get_additional(self) -> dict[str, Any]:
        """Per-run metadata reported by some harnesses. Empty by default."""
        return {}

    def get_memory_usage(self) -> float | None:
        """Return RSS in KB if ``resource`` is available, else ``None``."""
        try:
            import resource
        except ImportError:
            return None
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def done(self) -> None:
        """Release the index. Optional hook called by some harnesses."""
        self._index = None
        self._batch_results = None

    def _require_index(self) -> HNSWIndex:
        if self._index is None:
            msg = "index not built: call fit(X) first"
            raise RuntimeError(msg)
        return self._index

    def __str__(self) -> str:
        return f"vicinity-hnsw(M={self._m},ef={self._ef_search})"


class VicinityIVFPQ:
    """ann-benchmarks-compatible IVF-PQ wrapper around :class:`IVFPQIndex`."""

    def __init__(self, metric: str, method_param: Mapping[str, Any]) -> None:
        _require_cosine_metric(metric)
        self._num_clusters = int(cast(int, method_param.get("num_clusters", 256)))
        self._num_codebooks = _optional_int(method_param.get("num_codebooks"))
        self._codebook_size = int(cast(int, method_param.get("codebook_size", 256)))
        self._training_sample_size = _optional_int(
            method_param.get("training_sample_size")
        )
        self._kmeans_max_iter = int(cast(int, method_param.get("kmeans_max_iter", 100)))
        self._use_opq = bool(method_param.get("use_opq", False))
        self._seed = int(cast(int, method_param.get("seed", 42)))
        self._nprobe = int(cast(int, method_param.get("nprobe", 1)))
        self._rerank_pool = _normalize_rerank_pool(method_param.get("rerank_pool"))
        self._index: IVFPQIndex | None = None
        self._batch_results: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Build the index from training data."""
        if X.ndim != 2:
            msg = f"fit() expects a 2-D array, got shape {X.shape}"
            raise ValueError(msg)
        _n, dim = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)

        self._index = IVFPQIndex(
            dim=dim,
            num_clusters=self._num_clusters,
            num_codebooks=self._num_codebooks,
            codebook_size=self._codebook_size,
            nprobe=self._nprobe,
            use_opq=self._use_opq,
            seed=self._seed,
        )
        self._index.add_items(X)
        self._index.build(
            training_sample_size=self._training_sample_size,
            kmeans_max_iter=self._kmeans_max_iter,
        )

    def set_query_arguments(
        self,
        nprobe: int,
        rerank_pool: int | None = None,
    ) -> None:
        """Set nprobe and optional rerank pool for subsequent queries."""
        self._nprobe = int(nprobe)
        if rerank_pool is not None:
            self._rerank_pool = _normalize_rerank_pool(rerank_pool)
        if self._index is not None:
            self._index.set_nprobe(self._nprobe)

    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """Single-query interface (ann-benchmarks). Returns ids only."""
        index = self._require_index()
        q = np.ascontiguousarray(q, dtype=np.float32)
        ids, _dists = index.search(
            q,
            k=n,
            nprobe=self._nprobe,
            rerank_pool=self._rerank_pool,
        )
        return ids

    def batch_query(self, X: np.ndarray, n: int) -> None:
        """Batch-query interface (big-ann-benchmarks / VIBE)."""
        index = self._require_index()
        X = np.ascontiguousarray(X, dtype=np.float32)
        ids, _dists = index.batch_search(
            X,
            k=n,
            nprobe=self._nprobe,
            rerank_pool=self._rerank_pool,
        )
        self._batch_results = ids

    def get_batch_results(self) -> np.ndarray:
        """Return ids from the last :meth:`batch_query` call."""
        if self._batch_results is None:
            msg = "no batch results: call batch_query(...) first"
            raise RuntimeError(msg)
        return self._batch_results

    def get_additional(self) -> dict[str, Any]:
        """Per-run metadata reported by some harnesses. Empty by default."""
        return {}

    def get_memory_usage(self) -> float | None:
        """Return RSS in KB if ``resource`` is available, else ``None``."""
        try:
            import resource
        except ImportError:
            return None
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    def done(self) -> None:
        """Release the index. Optional hook called by some harnesses."""
        self._index = None
        self._batch_results = None

    def _require_index(self) -> IVFPQIndex:
        if self._index is None:
            msg = "index not built: call fit(X) first"
            raise RuntimeError(msg)
        return self._index

    def __str__(self) -> str:
        return (
            "vicinity-ivfpq("
            f"clusters={self._num_clusters},"
            f"codebooks={self._num_codebooks},"
            f"codebook_size={self._codebook_size},"
            f"nprobe={self._nprobe},"
            f"rerank_pool={self._rerank_pool})"
        )


def _parse_metric(metric: str) -> DistanceMetric:
    """Convert an ann-benchmarks metric string to a ``DistanceMetric`` enum."""
    m = metric.lower()
    if m in ("angular", "cosine"):
        return DistanceMetric.Cosine
    if m in ("euclidean", "l2"):
        return DistanceMetric.L2
    if m in ("ip", "inner", "inner_product", "dot"):
        return DistanceMetric.InnerProduct
    msg = f"unknown metric: {metric!r}"
    raise ValueError(msg)


def _require_cosine_metric(metric: str) -> None:
    if metric.lower() not in ("angular", "cosine"):
        msg = "IVF-PQ wrapper only supports angular/cosine metrics"
        raise ValueError(msg)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(cast(int, value))


def _normalize_rerank_pool(value: object) -> int | None:
    pool = _optional_int(value)
    if pool is None or pool <= 0:
        return None
    return pool
