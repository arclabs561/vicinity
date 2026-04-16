"""ann-benchmarks / big-ann-benchmarks BaseANN wrapper for vicinity.

Usage with ann-benchmarks:
    Copy or symlink this into ann_benchmarks/algorithms/vicinity/module.py
    and create the corresponding config.yml and Dockerfile.

Standalone usage:
    from vicinity.ann_benchmarks import VicinityHNSW
    algo = VicinityHNSW("cosine", {"M": 16, "efConstruction": 200})
    algo.fit(train_vectors)
    algo.set_query_arguments(100)  # ef_search
    algo.query(query_vector, 10)
"""

from __future__ import annotations

import numpy as np

from vicinity._vicinity import DistanceMetric, HNSWIndex


class VicinityHNSW:
    """ann-benchmarks-compatible HNSW wrapper.

    Conforms to the BaseANN interface expected by:
    - ann-benchmarks (erikbern/ann-benchmarks)
    - big-ann-benchmarks (harsha-simhadri/big-ann-benchmarks)
    - VIBE (vector-index-bench/vibe)
    """

    def __init__(self, metric: str, method_param: dict):
        self._metric_name = metric
        self._m = method_param.get("M", 16)
        self._ef_construction = method_param.get("efConstruction", 200)
        self._ef_search = 50
        self._index = None
        self._metric = _parse_metric(metric)
        self._results = None

    def fit(self, X: np.ndarray) -> None:
        """Build the index from training data."""
        _n, dim = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)

        # Angular/cosine metrics need auto-normalization for unnormalized inputs.
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
        self._ef_search = ef_search
        if self._index is not None:
            self._index.set_ef_search(ef_search)

    def query(self, q: np.ndarray, n: int) -> np.ndarray:
        """Single-query interface (ann-benchmarks)."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        ids, _dists = self._index.search(q, k=n, ef=self._ef_search)
        return ids

    def batch_query(self, X: np.ndarray, n: int) -> None:
        """Batch-query interface (big-ann-benchmarks / VIBE)."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        ids, _dists = self._index.batch_search(X, k=n, ef=self._ef_search)
        self._results = ids

    def get_batch_results(self) -> np.ndarray:
        """Return results from the last batch_query call."""
        return self._results

    def get_additional(self) -> dict:
        return {}

    def get_memory_usage(self) -> float | None:
        """Return RSS in KB (optional)."""
        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return None

    def __str__(self) -> str:
        return f"vicinity-hnsw(M={self._m},ef={self._ef_search})"


def _parse_metric(metric: str) -> DistanceMetric:
    """Convert ann-benchmarks metric string to vicinity enum."""
    m = metric.lower()
    if m in ("angular", "cosine"):
        return DistanceMetric.Cosine
    if m in ("euclidean", "l2"):
        return DistanceMetric.L2
    if m in ("ip", "inner", "inner_product", "dot"):
        return DistanceMetric.InnerProduct
    msg = f"unknown metric: {metric!r}"
    raise ValueError(msg)
