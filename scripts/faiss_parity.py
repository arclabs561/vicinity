#!/usr/bin/env python3
"""FAISS parity rows on the vicinity corpus and query set.

The harness writes both sides, so this script needs two files from
``examples/large_scale``: the dataset and the query set
(``--dump-queries``). Using the same queries matters: a second draw from the
same distribution is not the same query set, and recall moves with the sample.

Protocol, chosen to match the vicinity rows:

- Inner product on L2-normalised vectors, which is cosine distance up to the
  constant the harness also drops.
- One thread (``faiss.omp_set_num_threads(1)``). FAISS parallelises both build
  and search across the index; a multi-threaded row is not a comparison.
- Five untimed warm-up queries, then one timed pass over the whole query set.
- Ground truth is an exact top-k numpy scan of the same corpus, not FAISS's
  own, and it is verified against FAISS's exact flat index before any ANN row is
  written.

Examples::

    # Build both FAISS families on the 100k corpus and write rows.
    python scripts/faiss_parity.py \
      --data "$ANN_DATA_ROOT/datasets/uniform-n100000-d96.bin" \
      --queries /tmp/queries-100k.bin \
      --out benchmarks/faiss-parity-100k.jsonl

    # HNSW only, one ef sweep.
    python scripts/faiss_parity.py --data ... --queries ... --families faiss-hnsw
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

MAGIC = b"VICDS01\0"
HEADER = 20


def read_vectors(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise SystemExit(f"{path}: magic {raw[:8]!r}, expected {MAGIC!r}")
    dim = int.from_bytes(raw[8:12], "little")
    count = int.from_bytes(raw[12:20], "little")
    values = np.frombuffer(raw[HEADER:], dtype="<f4")
    if values.size != dim * count:
        raise SystemExit(f"{path}: {values.size} floats, expected {dim * count}")
    return values.reshape(count, dim).astype(np.float32)


def load_average() -> float | None:
    try:
        text = Path("/proc/loadavg").read_text()
        return float(text.split()[0])
    except OSError:
        pass
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True, check=True
        ).stdout
        return float(out.strip().strip("{}").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def exact_top_k(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact top-k inner-product ids, one query at a time to bound memory."""
    truth = np.empty((len(queries), k), dtype=np.int64)
    for index, query in enumerate(queries):
        scores = corpus @ query
        truth[index] = np.argpartition(-scores, k)[:k]
        order = np.argsort(-scores[truth[index]])
        truth[index] = truth[index][order]
    return truth


def measure(search, queries: np.ndarray, truth: np.ndarray, k: int, repeat: int = 1):
    """Warm up, then `repeat` timed passes.

    Returns (recall, p50, p95, p99, qps). Recall is identical across passes at a
    fixed index, so it is taken from the last one; the quantiles are the median
    across passes, which is what a single pass cannot give: five tail
    observations at 500 queries make a one-pass p99 unstable.
    """
    for query in queries[:5]:
        search(query, k)
    passes: list[tuple[float, np.ndarray, int]] = []
    for _ in range(max(1, repeat)):
        latencies = np.empty(len(queries), dtype=np.float64)
        hits = 0
        for index, query in enumerate(queries):
            start = time.perf_counter()
            _, ids = search(query, k)
            ids = np.asarray(ids).reshape(-1)
            latencies[index] = (time.perf_counter() - start) * 1e6
            hits += len(set(ids.tolist()) & set(truth[index].tolist()))
        passes.append((hits / (len(queries) * k), np.sort(latencies), hits))
    recall = passes[-1][0]
    p50 = float(np.median([float(np.percentile(lat, 50)) for _, lat, _ in passes]))
    p95 = float(np.median([float(np.percentile(lat, 95)) for _, lat, _ in passes]))
    p99 = float(np.median([float(np.percentile(lat, 99)) for _, lat, _ in passes]))
    qps = float(np.median([len(queries) / (lat.sum() / 1e6) for _, lat, _ in passes]))
    return recall, p50, p95, p99, qps


def call_floor_us(faiss, queries: np.ndarray, dim: int, k: int) -> float:
    """Per-call Python + FFI floor: the same call against a one-vector index.

    Every FAISS row is measured through one Python call per query, so the row
    carries this floor. Recording it lets a reader see how much of a 100 us row
    is the call rather than the search.
    """
    tiny = faiss.IndexFlatIP(dim)
    tiny.add(np.zeros((1, dim), dtype=np.float32))
    samples = []
    for query in queries:
        start = time.perf_counter()
        tiny.search(query.reshape(1, -1), k)
        samples.append((time.perf_counter() - start) * 1e6)
    return float(np.median(samples))


def row(
    family: str, knob: str, value: int, stats, build_s: float, extra: dict, meta: dict
):
    recall, p50, p95, p99, qps = stats
    entry = {
        "family": family,
        "storage_mode": "resident",
        "dataset": meta["dataset"],
        "knob": knob,
        "knob_value": value,
        "recall": round(recall, 4),
        "recall_at_10": round(recall, 4),
        "p50_us": round(p50, 1),
        "p50_us_minus_floor": round(max(p50 - meta.get("call_floor_us", 0.0), 0.0), 1),
        "p95_us": round(p95, 1),
        "p99_us": round(p99, 1),
        "qps": round(qps, 1),
        "dists_per_query": None,
        "build_s": round(build_s, 4),
        "mem_mib": None,
        "ratio": round(meta["exact_p50_us"] / max(p50, 1e-9), 4),
        "inverse_ratio": round(meta["exact_p50_us"] / max(p50, 1e-9), 4),
        "n": meta["n"],
        "dim": meta["dim"],
        "gt_verified": True,
        "errors": 0,
        **extra,
    }
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef", default="10,20,50,100,200,400")
    parser.add_argument("--nlist", default="64,256")
    parser.add_argument("--m-subq", default="8,16")
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--nprobe", default="1,2,4,8,16")
    parser.add_argument(
        "--families", default="faiss-hnsw,faiss-ivfpq", help="comma-separated subset"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="timed passes per knob value; quantiles are the median across passes",
    )
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    args = parser.parse_args(argv)

    import faiss

    faiss.omp_set_num_threads(1)

    corpus = read_vectors(args.data)
    queries = read_vectors(args.queries)
    n, dim = corpus.shape
    if queries.shape[1] != dim:
        raise SystemExit(f"query dim {queries.shape[1]} != corpus dim {dim}")
    families = [name.strip() for name in args.families.split(",") if name.strip()]

    print(f"corpus {n} x {dim}, queries {len(queries)}, k={args.k}", file=sys.stderr)
    truth = exact_top_k(corpus, queries, args.k)

    flat = faiss.IndexFlatIP(dim)
    flat.add(corpus)
    check = 0
    for index, query in enumerate(queries):
        _, ids = flat.search(query.reshape(1, -1), args.k)
        check += len(set(ids[0].tolist()) & set(truth[index].tolist()))
    coverage = check / (len(queries) * args.k)
    if coverage < 0.999:
        raise SystemExit(
            f"ground truth disagrees with FAISS's exact index ({coverage:.4f}); refusing to "
            "write ANN rows against it"
        )

    exact_latencies = []
    for query in queries:
        start = time.perf_counter()
        flat.search(query.reshape(1, -1), args.k)
        exact_latencies.append((time.perf_counter() - start) * 1e6)
    exact_p50 = float(np.percentile(exact_latencies, 50))

    floor = call_floor_us(faiss, queries, dim, args.k)
    print(f"call floor: {floor:.1f} us per query", file=sys.stderr)
    dataset_label = f"uniform@{n // 1000}k" if n % 1000 == 0 else f"uniform@{n}"
    meta = {
        "_meta": True,
        "schema": 1,
        "harness": "faiss_parity",
        "host": platform.node(),
        "cores": faiss.omp_get_max_threads(),
        "load1": load_average(),
        "n": int(n),
        "dim": int(dim),
        "k": args.k,
        "queries": int(len(queries)),
        "exact_queries": int(len(queries)),
        "dataset": dataset_label,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "call_floor_us": round(floor, 1),
        "repeat": max(1, args.repeat),
        "caveat": (
            "call_floor_us is the same Python call against a one-vector index. Every "
            "FAISS latency here includes it, so p50_us_minus_floor estimates the search "
            "alone; the raw p50 is what the call actually costs a caller."
        ),
        "exact_p50_us": round(exact_p50, 1),
        "metric": "inner product on L2-normalised vectors",
        "threads": 1,
    }
    rows: list[dict] = []

    if "faiss-hnsw" in families:
        index = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = args.hnsw_ef_construction
        start = time.perf_counter()
        index.add(corpus)
        build_s = time.perf_counter() - start
        for ef in (int(value) for value in args.ef.split(",")):
            index.hnsw.efSearch = ef
            stats = measure(
                lambda query, k, index=index: index.search(query.reshape(1, -1), k)[
                    0:2
                ],
                queries,
                truth,
                args.k,
                args.repeat,
            )
            rows.append(
                row(
                    "faiss-hnsw",
                    "ef_search",
                    ef,
                    stats,
                    build_s,
                    {
                        "build_m": args.hnsw_m,
                        "build_ef_construction": args.hnsw_ef_construction,
                    },
                    meta,
                )
            )
            print(json.dumps(rows[-1]), flush=True)
        del index

    if "faiss-ivfpq" in families:
        effective_m = [
            value
            for value in (int(v) for v in args.m_subq.split(","))
            if dim % value == 0
        ]
        if not effective_m:
            raise SystemExit(f"--m-subq must divide dim {dim}")
        for nlist in (int(value) for value in args.nlist.split(",")):
            for m in effective_m:
                index = faiss.IndexIVFPQ(
                    faiss.IndexFlatIP(dim),
                    dim,
                    nlist,
                    m,
                    args.nbits,
                    faiss.METRIC_INNER_PRODUCT,
                )
                start = time.perf_counter()
                # FAISS wants at least `nlist` training points per centroid to
                # learn a codebook; the corpus is used when it is small.
                train_rows = min(n, max(50 * nlist, 10_000))
                index.train(corpus[:train_rows])
                trained_s = time.perf_counter() - start
                index.add(corpus)
                build_s = time.perf_counter() - start
                for nprobe in (int(value) for value in args.nprobe.split(",")):
                    index.nprobe = nprobe
                    stats = measure(
                        lambda query, k, index=index: index.search(
                            query.reshape(1, -1), k
                        )[0:2],
                        queries,
                        truth,
                        args.k,
                        args.repeat,
                    )
                    rows.append(
                        row(
                            "faiss-ivfpq",
                            "nprobe",
                            nprobe,
                            stats,
                            build_s,
                            {
                                "nlist": nlist,
                                "m_subq": m,
                                "nbits": args.nbits,
                                "trained_s": round(trained_s, 4),
                                "stage": "resident",
                            },
                            meta,
                        )
                    )
                    print(json.dumps(rows[-1]), flush=True)
                del index

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
        for entry in rows:
            handle.write(json.dumps(entry) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
