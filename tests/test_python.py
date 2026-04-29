"""End-to-end tests for the pyvicinity Python bindings.

Run with::

    .venv/bin/maturin develop --release
    .venv/bin/python -m pytest tests/test_python.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from pyvicinity import (
    MISSING_DISTANCE,
    MISSING_LABEL,
    DistanceMetric,
    HNSWIndex,
    __version__,
)


def _build(
    n: int = 200,
    dim: int = 16,
    *,
    metric: DistanceMetric = DistanceMetric.Cosine,
    auto_normalize: bool | None = None,
    seed: int = 0,
) -> tuple[HNSWIndex, np.ndarray]:
    """Build a small index with sensible defaults for the metric."""
    if auto_normalize is None:
        auto_normalize = metric in (DistanceMetric.Cosine, DistanceMetric.Angular)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, dim), dtype=np.float32)
    idx = HNSWIndex(
        dim=dim,
        metric=metric,
        auto_normalize=auto_normalize,
        seed=seed,
    )
    idx.add_items(X)
    idx.build()
    return idx, X


def test_version_is_exposed() -> None:
    assert isinstance(__version__, str)
    assert __version__.count(".") >= 2  # semver-shaped


def test_len_and_getters() -> None:
    idx, _ = _build(n=50, dim=8)
    assert len(idx) == 50
    assert idx.num_vectors == 50
    assert idx.dimension == 8
    assert idx.m == 16
    assert idx.ef_construction == 200
    assert idx.ef_search == 50
    assert idx.metric == DistanceMetric.Cosine
    assert idx.auto_normalize is True


def test_repr_is_pythonic() -> None:
    idx, _ = _build(n=10, dim=4)
    r = repr(idx)
    assert r.startswith("HNSWIndex(")
    assert "metric=DistanceMetric.Cosine" in r
    assert "auto_normalize=True" in r  # not Rust's `true`


@pytest.mark.parametrize(
    ("metric", "auto_normalize", "expect_zero"),
    [
        (DistanceMetric.L2, False, True),
        (DistanceMetric.Cosine, True, True),
        (DistanceMetric.Angular, True, True),
        (DistanceMetric.Angular, False, True),  # angular handles norms itself
        (DistanceMetric.InnerProduct, False, False),  # -||x||^2, not zero
    ],
)
def test_self_search_returns_self(
    metric: DistanceMetric, auto_normalize: bool, expect_zero: bool
) -> None:
    idx, X = _build(n=200, dim=16, metric=metric, auto_normalize=auto_normalize)
    ids, dists = idx.search(X[0], k=5)
    assert ids[0] == 0, f"top-1 should be the query itself for {metric}"
    if expect_zero:
        assert abs(float(dists[0])) < 1e-4, (
            f"top-1 self-distance should be ~0 for {metric} "
            f"(auto_normalize={auto_normalize}), got {dists[0]}"
        )


def test_auto_normalize_symmetric_for_angular() -> None:
    """auto_normalize=True must apply to BOTH insert AND query for Angular,
    not just insert. Regression for the asymmetric prep_query branch.

    Setup: a tiny ANN index with one inserted unit vector, and a query that's
    the same direction but a much larger magnitude. With symmetric
    normalization, top-1 distance should be ~0.
    """
    rng = np.random.default_rng(0)
    base = rng.standard_normal(8, dtype=np.float32)
    base /= np.linalg.norm(base)

    idx = HNSWIndex(dim=8, metric=DistanceMetric.Angular, auto_normalize=True, seed=0)
    idx.add_items(np.stack([base, -base, rng.standard_normal(8).astype(np.float32)]))
    idx.build()

    scaled_query = (base * 137.5).astype(np.float32)
    ids, dists = idx.search(scaled_query, k=1)
    assert ids[0] == 0
    assert abs(float(dists[0])) < 1e-4, (
        f"Angular self-distance with scaled query should be ~0, got {dists[0]}"
    )


def test_auto_normalize_rejected_for_l2() -> None:
    """auto_normalize=True with L2/InnerProduct silently distorts distances;
    the binding should reject the combo at construction time."""
    with pytest.raises(ValueError, match="auto_normalize"):
        HNSWIndex(dim=4, metric=DistanceMetric.L2, auto_normalize=True)
    with pytest.raises(ValueError, match="auto_normalize"):
        HNSWIndex(dim=4, metric=DistanceMetric.InnerProduct, auto_normalize=True)


def test_dtypes() -> None:
    idx, X = _build()
    ids, dists = idx.search(X[0], k=5)
    assert ids.dtype == np.int64
    assert dists.dtype == np.float32
    bids, bdists = idx.batch_search(X[:3], k=5)
    assert bids.dtype == np.int64
    assert bdists.dtype == np.float32


def test_batch_shape_and_padding() -> None:
    idx, X = _build()
    ids, dists = idx.batch_search(X[:7], k=10)
    assert ids.shape == (7, 10)
    assert dists.shape == (7, 10)


def test_batch_padding_when_k_exceeds_n() -> None:
    """Rows shorter than k are padded with MISSING_LABEL / MISSING_DISTANCE."""
    idx, _ = _build(n=5, dim=4)
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((2, 4), dtype=np.float32)
    ids, dists = idx.batch_search(Q, k=8)
    assert ids.shape == (2, 8)
    # First 5 columns should be valid; last 3 are sentinel.
    assert np.all(ids[:, :5] < 5)
    assert np.all(ids[:, 5:] == MISSING_LABEL)
    assert np.all(dists[:, 5:] == MISSING_DISTANCE)


def test_missing_sentinels_are_canonical() -> None:
    """MISSING_LABEL is -1 (faiss convention) and MISSING_DISTANCE is +inf."""
    assert MISSING_LABEL == -1
    assert float("inf") == MISSING_DISTANCE


def test_dimension_mismatch_raises_valueerror() -> None:
    idx, _ = _build(n=20, dim=8)
    with pytest.raises(ValueError, match="dimension"):
        idx.search(np.zeros(7, dtype=np.float32), k=3)
    with pytest.raises(ValueError, match="dimension"):
        idx.batch_search(np.zeros((2, 7), dtype=np.float32), k=3)


def test_explicit_ids_round_trip() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4), dtype=np.float32)
    ids = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.int64)
    idx = HNSWIndex(dim=4, metric=DistanceMetric.L2, seed=0)
    idx.add_items(X, ids=ids)
    idx.build()
    found, _ = idx.search(X[3], k=1)
    assert found[0] == 103


def test_id_length_mismatch_raises() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4), dtype=np.float32)
    bad_ids = np.array([1, 2, 3], dtype=np.int64)
    idx = HNSWIndex(dim=4, metric=DistanceMetric.L2)
    with pytest.raises(ValueError, match="ids length"):
        idx.add_items(X, ids=bad_ids)


def test_negative_id_rejected() -> None:
    """ids must be in [0, 2**32); negatives raise ValueError."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((3, 4), dtype=np.float32)
    bad_ids = np.array([0, -1, 2], dtype=np.int64)
    idx = HNSWIndex(dim=4, metric=DistanceMetric.L2)
    with pytest.raises(ValueError, match="out of range"):
        idx.add_items(X, ids=bad_ids)


def test_id_too_large_rejected() -> None:
    """ids > u32::MAX raise ValueError; pyvicinity stores u32 internally."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2, 4), dtype=np.float32)
    bad_ids = np.array([0, 1 << 33], dtype=np.int64)
    idx = HNSWIndex(dim=4, metric=DistanceMetric.L2)
    with pytest.raises(ValueError, match="out of range"):
        idx.add_items(X, ids=bad_ids)


def test_set_ef_search_sticks() -> None:
    idx, _ = _build()
    assert idx.ef_search == 50
    idx.set_ef_search(123)
    assert idx.ef_search == 123


def test_distance_metric_equality() -> None:
    assert DistanceMetric.Cosine == DistanceMetric.Cosine
    assert DistanceMetric.Cosine != DistanceMetric.L2


def test_recall_against_brute_force() -> None:
    """Mean recall@10 must be high at ef=100 on a small Cosine corpus.

    The single test that catches a regression in actual ANN behavior
    (vs the plumbing tests above which would still pass even if the
    index returned arbitrary IDs in the right shape).

    Setup: 2000 unit-norm vectors in dim 32. Brute-force top-10 by
    dot product gives ground truth; pyvicinity's HNSW must recover
    >=95% of those. Probed at this seed: actual mean is 1.000 with
    min 1.000, so 0.95 leaves wide slack for HNSW build variance.
    """
    rng = np.random.default_rng(42)
    n, dim, k, nq = 2_000, 32, 10, 50

    corpus = rng.standard_normal((n, dim), dtype=np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    sims = queries @ corpus.T
    truth = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    truth_sets = [set(row.tolist()) for row in truth]

    idx = HNSWIndex(
        dim=dim,
        m=16,
        ef_construction=100,
        metric=DistanceMetric.Cosine,
        seed=1,
    )
    idx.add_items(corpus)
    idx.build()
    idx.set_ef_search(100)

    ann_ids, _ = idx.batch_search(queries, k=k)
    recalls = [
        len(set(ann_ids[i].tolist()) & truth_sets[i]) / k for i in range(nq)
    ]
    mean_recall = float(np.mean(recalls))
    assert mean_recall >= 0.95, f"mean recall@{k} = {mean_recall:.3f} < 0.95"


def test_search_before_build_raises() -> None:
    """search() on an unbuilt index must raise, not return garbage."""
    rng = np.random.default_rng(0)
    idx = HNSWIndex(dim=8, metric=DistanceMetric.L2, seed=0)
    idx.add_items(rng.standard_normal((5, 8), dtype=np.float32))
    # Note: build() not called.
    with pytest.raises(ValueError, match="must be built"):
        idx.search(np.zeros(8, dtype=np.float32), k=3)


def test_empty_index_build_raises() -> None:
    """build() on an index with no add_items() must raise, not silently succeed."""
    idx = HNSWIndex(dim=8, metric=DistanceMetric.L2, seed=0)
    with pytest.raises(ValueError, match="empty"):
        idx.build()


def test_seed_reproducibility() -> None:
    """Same seed + same data should give identical search results."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 16), dtype=np.float32)

    def run() -> tuple[np.ndarray, np.ndarray]:
        i = HNSWIndex(dim=16, metric=DistanceMetric.L2, seed=42)
        i.add_items(X)
        i.build()
        return i.search(X[0], k=10)

    a_ids, a_d = run()
    b_ids, b_d = run()
    np.testing.assert_array_equal(a_ids, b_ids)
    np.testing.assert_allclose(a_d, b_d, rtol=1e-6)


def test_ann_benchmarks_wrapper_smoke() -> None:
    from pyvicinity.ann_benchmarks import VicinityHNSW

    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 8), dtype=np.float32)

    algo = VicinityHNSW("cosine", {"M": 8, "efConstruction": 50})
    algo.fit(X)
    algo.set_query_arguments(20)

    ids = algo.query(X[0], 5)
    assert ids.shape == (5,)
    assert ids[0] == 0

    algo.batch_query(X[:4], 3)
    batch = algo.get_batch_results()
    assert batch.shape == (4, 3)


def test_ann_benchmarks_unfit_raises() -> None:
    from pyvicinity.ann_benchmarks import VicinityHNSW

    algo = VicinityHNSW("l2", {})
    with pytest.raises(RuntimeError, match="fit"):
        algo.query(np.zeros(4, dtype=np.float32), 1)
    with pytest.raises(RuntimeError, match="batch_query"):
        algo.get_batch_results()


def test_ann_benchmarks_unknown_metric_raises() -> None:
    from pyvicinity.ann_benchmarks import VicinityHNSW

    with pytest.raises(ValueError, match="unknown metric"):
        VicinityHNSW("hamming", {})
