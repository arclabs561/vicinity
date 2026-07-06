# /// script
# requires-python = ">=3.10"
# dependencies = ["pyvicinity", "numpy"]
# ///
"""How do I tune ef_search? Sweep it and measure recall@10.

The standard recall sweep every ANN user runs once: build an index, vary
``ef_search``, compare ANN top-k against brute-force ground truth. The
output is the curve users actually look at when choosing parameters.

Synthetic unit-norm vectors here so the example runs in seconds with no
external download. Real cosine-embedding distributions (sentence
embeddings, image features, etc.) behave qualitatively the same -- ef
controls the speed/recall tradeoff and the curve shape is similar.

Run with:

    uv run examples/python/02_batch_and_recall.py
"""

from __future__ import annotations

import time

import numpy as np

from pyvicinity import MISSING_LABEL, DistanceMetric, HNSWIndex


def brute_force_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    sims = queries @ corpus.T
    return np.argpartition(-sims, kth=k, axis=1)[:, :k]


def main() -> None:
    rng = np.random.default_rng(0)
    n, nq, dim, k = 50_000, 500, 64, 10

    print(f"corpus: n={n} dim={dim}  queries: {nq}  k={k}")
    corpus = rng.standard_normal((n, dim), dtype=np.float32)
    corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
    queries = rng.standard_normal((nq, dim), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    t0 = time.perf_counter()
    truth = brute_force_topk(corpus, queries, k)
    truth_sets = [set(row.tolist()) for row in truth]
    print(f"brute-force ground truth: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    index = HNSWIndex(
        dim=dim, m=32, ef_construction=200, metric=DistanceMetric.Cosine, seed=1
    )
    index.add_items(corpus)
    index.build()
    print(f"index build:              {time.perf_counter() - t0:.2f}s\n")

    print(f"{'ef_search':>10} {'recall@k':>10} {'qps':>10}")
    for ef in (10, 25, 50, 100, 200, 400):
        index.set_ef_search(ef)
        t0 = time.perf_counter()
        ann_ids, _ = index.batch_search(queries, k=k)
        elapsed = time.perf_counter() - t0

        valid_mask = ann_ids != MISSING_LABEL
        recalls = []
        for i in range(nq):
            row = ann_ids[i][valid_mask[i]]
            recalls.append(len(set(row.tolist()) & truth_sets[i]) / k)

        print(f"{ef:10d} {np.mean(recalls):10.3f} {nq / elapsed:10.1f}")


if __name__ == "__main__":
    main()
