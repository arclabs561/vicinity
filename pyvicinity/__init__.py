"""pyvicinity: multi-algorithm approximate nearest-neighbor search (Rust-backed).

Python bindings for the ``vicinity`` Rust crate (crates.io). The PyPI
distribution is named ``pyvicinity`` because the bare ``vicinity`` name
is held on PyPI by an unrelated ANN project.

Example::

    import numpy as np
    from pyvicinity import HNSWIndex, DistanceMetric

    index = HNSWIndex(dim=128, metric=DistanceMetric.Cosine)
    index.add_items(np.random.randn(10000, 128).astype(np.float32))
    index.build()
    ids, dists = index.search(query, k=10, ef=50)
"""

from pyvicinity._core import DistanceMetric, HNSWIndex

__all__ = ["HNSWIndex", "DistanceMetric"]
