# /// script
# requires-python = ">=3.10"
# dependencies = ["pyvicinity", "numpy"]
# ///
"""Drop pyvicinity into the ann-benchmarks harness contract.

Anyone benchmarking ANN libraries against ann-benchmarks /
big-ann-benchmarks / VIBE expects a class with ``fit``, ``query``, and
``batch_query`` methods. ``pyvicinity.ann_benchmarks.VicinityHNSW``
implements that contract verbatim. This example exercises the full
harness flow against a 5k synthetic corpus so you can drop the wrapper
into the harness with confidence.

Run with:

    uv run examples/python/03_ann_benchmarks_harness.py
"""

from __future__ import annotations

import numpy as np

from pyvicinity.ann_benchmarks import VicinityHNSW


def main() -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((5_000, 32), dtype=np.float32)
    test = rng.standard_normal((50, 32), dtype=np.float32)

    algo = VicinityHNSW("cosine", {"M": 16, "efConstruction": 100})

    # 1. fit() -- the harness builds the index once.
    algo.fit(train)
    print(f"fit ok:  {algo}")

    # 2. set_query_arguments() -- one ef_search value per recall point on
    #    the harness's recall/QPS plot.
    algo.set_query_arguments(50)

    # 3. query() -- single-query path used by erikbern/ann-benchmarks.
    ids = algo.query(test[0], 10)
    print(f"single-query ids[:10]: {ids.tolist()}")

    # 4. batch_query() -- preferred path for big-ann-benchmarks and VIBE.
    algo.batch_query(test, 10)
    batch = algo.get_batch_results()
    print(f"batch_query shape:     {batch.shape}")

    # 5. done() -- some harnesses release the index between configs.
    algo.done()


if __name__ == "__main__":
    main()
