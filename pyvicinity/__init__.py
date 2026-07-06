"""pyvicinity: HNSW approximate nearest-neighbor search.

Python bindings for the ``vicinity`` Rust crate (crates.io). The PyPI
distribution is named ``pyvicinity`` because the bare ``vicinity`` name
is held on PyPI by an unrelated project. The current Python API exposes HNSW;
other vicinity algorithms are Rust-only for now.

Quick start::

    import numpy as np
    from pyvicinity import HNSWIndex, DistanceMetric

    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10_000, 128), dtype=np.float32)

    index = HNSWIndex(
        dim=128,
        metric=DistanceMetric.Cosine,
        auto_normalize=True,  # normalizes both inserts and queries
        seed=42,              # reproducible build order
    )
    index.add_items(vectors)
    index.build()

    ids, dists = index.search(vectors[0], k=10)
    batch_ids, batch_dists = index.batch_search(vectors[:32], k=10)
"""

from pyvicinity._core import (
    MISSING_DISTANCE,
    MISSING_LABEL,
    DistanceMetric,
    HNSWIndex,
)
from pyvicinity._core import __version__ as __version__

__all__ = [
    "MISSING_DISTANCE",
    "MISSING_LABEL",
    "DistanceMetric",
    "HNSWIndex",
    "__version__",
]
