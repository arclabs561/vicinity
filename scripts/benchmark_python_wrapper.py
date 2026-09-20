#!/usr/bin/env python3
"""Measure pyvicinity end-to-end and native-adjacent Python costs.

The output is JSONL so it can be compared with the Rust ANN harness without
pretending that Python call, NumPy conversion, and native search are one cost.
Build the extension first, for example with ``maturin develop --release``.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from pyvicinity import DistanceMetric, HNSWIndex


def rss_kb() -> int:
    """Return process peak RSS in KB on the supported Unix runners."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value / 1024) if sys.platform == "darwin" else int(value)


def timed(call: Any) -> tuple[Any, float]:
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1_000_000_000


def metric_value(name: str) -> DistanceMetric:
    try:
        return {
            "cosine": DistanceMetric.Cosine,
            "angular": DistanceMetric.Angular,
            "l2": DistanceMetric.L2,
            "inner_product": DistanceMetric.InnerProduct,
        }[name]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unsupported metric: {name}") from error


def benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    vectors = rng.standard_normal((args.train, args.dim), dtype=np.float32)
    queries = rng.standard_normal((args.queries, args.dim), dtype=np.float32)
    if args.metric in {"cosine", "angular"}:
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    index = HNSWIndex(
        dim=args.dim,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        metric=metric_value(args.metric),
        auto_normalize=False,
        seed=args.seed,
    )
    _, build_s = timed(lambda: (index.add_items(vectors), index.build()))

    warm_queries = queries[: min(args.warmup, args.queries)]
    index.batch_search(warm_queries, args.k, args.ef_search)
    single_times: list[float] = []
    for query in queries:
        _, elapsed = timed(lambda query=query: index.search(query, args.k, args.ef_search))
        single_times.append(elapsed)

    batch_times: list[float] = []
    for start in range(0, args.queries, args.batch_size):
        batch = queries[start : start + args.batch_size]
        _, elapsed = timed(lambda batch=batch: index.batch_search(batch, args.k, args.ef_search))
        batch_times.append(elapsed)

    noncontiguous = queries[:, ::1]
    _, copy_s = timed(lambda: np.ascontiguousarray(noncontiguous, dtype=np.float32))
    metadata = {
        "result_schema": 1,
        "algorithm": "hnsw",
        "metric": args.metric,
        "train": args.train,
        "queries": args.queries,
        "dim": args.dim,
        "k": args.k,
        "batch_size": args.batch_size,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "seed": args.seed,
        "warmup_queries": len(warm_queries),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "rss_kb_peak": rss_kb(),
    }
    return [
        {**metadata, "phase": "build", "seconds": build_s},
        {
            **metadata,
            "phase": "single_query",
            "queries_performed": len(single_times),
            "seconds_total": sum(single_times),
            "seconds_per_query_p50": float(np.percentile(single_times, 50)),
            "seconds_per_query_p95": float(np.percentile(single_times, 95)),
            "seconds_per_query_p99": float(np.percentile(single_times, 99)),
        },
        {
            **metadata,
            "phase": "batch_query",
            "batches": len(batch_times),
            "seconds_total": sum(batch_times),
            "seconds_per_batch_p50": float(np.percentile(batch_times, 50)),
            "seconds_per_batch_p95": float(np.percentile(batch_times, 95)),
        },
        {
            **metadata,
            "phase": "numpy_contiguous_conversion",
            "seconds": copy_s,
            "note": "The current fixture is contiguous; this records the conversion boundary separately.",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSONL here instead of stdout")
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=500)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=("cosine", "angular", "l2", "inner_product"), default="cosine")
    args = parser.parse_args()
    if min(args.train, args.queries, args.dim, args.k, args.batch_size, args.m, args.ef_construction, args.ef_search) <= 0:
        parser.error("sizes and search parameters must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = benchmark(args)
    output = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
