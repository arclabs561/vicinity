# /// script
# requires-python = ">=3.10"
# dependencies = ["pyvicinity", "numpy"]
# ///
"""Drop pyvicinity into the ann-benchmarks harness contract.

Anyone benchmarking ANN libraries against ann-benchmarks /
big-ann-benchmarks / VIBE expects a class with ``fit``, ``query``, and
``batch_query`` methods. ``pyvicinity.ann_benchmarks`` provides HNSW and
IVF-PQ wrappers for that contract. This example exercises the full harness
flow against a 5k synthetic corpus.

Run with:

    uv run examples/python/03_ann_benchmarks_harness.py
"""

from __future__ import annotations

import numpy as np

from pyvicinity.ann_benchmarks import VicinityHNSW, VicinityIVFPQ


def run_harness_flow(name: str, algo, train: np.ndarray, test: np.ndarray) -> None:
    # 1. fit() -- the harness builds the index once.
    algo.fit(train)
    print(f"{name} fit ok:  {algo}")

    # 2. set_query_arguments() -- one search parameter value per recall point.
    if isinstance(algo, VicinityIVFPQ):
        algo.set_query_arguments(8, rerank_pool=100)
    else:
        algo.set_query_arguments(50)

    # 3. query() -- single-query path used by erikbern/ann-benchmarks.
    ids = algo.query(test[0], 10)
    print(f"{name} single-query ids[:10]: {ids.tolist()}")

    # 4. batch_query() -- preferred path for big-ann-benchmarks and VIBE.
    algo.batch_query(test, 10)
    batch = algo.get_batch_results()
    print(f"{name} batch_query shape:     {batch.shape}")

    # 5. done() -- some harnesses release the index between configs.
    algo.done()


def main() -> None:
    rng = np.random.default_rng(0)
    train = rng.standard_normal((5_000, 32), dtype=np.float32)
    test = rng.standard_normal((50, 32), dtype=np.float32)

    run_harness_flow(
        "hnsw", VicinityHNSW("cosine", {"M": 16, "efConstruction": 100}), train, test
    )
    run_harness_flow(
        "ivfpq",
        VicinityIVFPQ(
            "cosine",
            {
                "num_clusters": 32,
                "num_codebooks": 8,
                "codebook_size": 32,
                "training_sample_size": 1_000,
                "kmeans_max_iter": 5,
                "nprobe": 8,
            },
        ),
        train,
        test,
    )


if __name__ == "__main__":
    main()
