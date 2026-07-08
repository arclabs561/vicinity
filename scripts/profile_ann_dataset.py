#!/usr/bin/env python3
"""Profile VEC1/NBR1 benchmark dataset shape and difficulty."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any, Literal

import numpy as np

Metric = Literal["cosine", "l2"]


def binary_shape(path: Path, magic: bytes) -> tuple[int, int]:
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except FileNotFoundError as exc:
        raise SystemExit(f"missing {path}") from exc
    if len(header) != 12 or header[:4] != magic:
        raise SystemExit(f"{path} is not a {magic.decode()} file")
    rows, width = struct.unpack("<II", header[4:])
    expected_size = 12 + rows * width * 4
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"{path} has {actual_size} bytes, expected {expected_size} "
            f"for shape ({rows}, {width})"
        )
    return rows, width


def read_vec1(path: Path) -> np.memmap:
    rows, dim = binary_shape(path, b"VEC1")
    return np.memmap(path, dtype=np.float32, mode="r", offset=12, shape=(rows, dim))


def read_nbr1(path: Path) -> np.memmap:
    rows, k = binary_shape(path, b"NBR1")
    return np.memmap(path, dtype=np.int32, mode="r", offset=12, shape=(rows, k))


def label_count(path: Path) -> int:
    try:
        with path.open("rb") as f:
            header = f.read(8)
    except FileNotFoundError as exc:
        raise SystemExit(f"missing {path}") from exc
    if len(header) != 8 or header[:4] != b"LBL1":
        raise SystemExit(f"{path} is not a LBL1 file")
    (rows,) = struct.unpack("<I", header[4:])
    expected_size = 8 + rows * 4
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"{path} has {actual_size} bytes, expected {expected_size} "
            f"for {rows} labels"
        )
    return rows


def quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def sample_indices(n: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    size = min(n, limit)
    if size == n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=size, replace=False))


def infer_metric(dataset: Path, requested: str) -> Metric:
    if requested in {"cosine", "l2"}:
        return requested  # type: ignore[return-value]
    manifest = dataset / "dataset.json"
    if manifest.exists():
        try:
            settings = json.loads(manifest.read_text(encoding="utf-8")).get(
                "settings",
                {},
            )
        except json.JSONDecodeError:
            settings = {}
        metric = settings.get("metric")
        if metric == "angular":
            return "cosine"
        if metric == "euclidean":
            return "l2"
    name = dataset.name.lower()
    if "euclidean" in name:
        return "l2"
    return "cosine"


def optional_split(
    dataset: Path,
    *,
    name: str,
    kind: str,
    query_file: str,
    neighbor_file: str,
    label_file: str | None = None,
) -> dict[str, Any] | None:
    query_path = dataset / query_file
    neighbor_path = dataset / neighbor_file
    if not query_path.exists() and not neighbor_path.exists():
        return None
    queries, dim = binary_shape(query_path, b"VEC1")
    neighbor_queries, k = binary_shape(neighbor_path, b"NBR1")
    if queries != neighbor_queries:
        raise SystemExit(
            f"{query_file} query count {queries} does not match "
            f"{neighbor_file} count {neighbor_queries}"
        )
    row: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "queries": queries,
        "dim": dim,
        "ground_truth_k": k,
    }
    if label_file is not None and (dataset / label_file).exists():
        labels = label_count(dataset / label_file)
        if labels != queries:
            raise SystemExit(
                f"{label_file} label count {labels} does not match {queries}"
            )
        row["label_file"] = label_file
    return row


def query_splits(
    dataset: Path, test_shape: tuple[int, int], neighbor_k: int
) -> list[dict[str, Any]]:
    rows = [
        {
            "name": "base",
            "kind": "in_distribution",
            "queries": test_shape[0],
            "dim": test_shape[1],
            "ground_truth_k": neighbor_k,
        }
    ]
    for split in (
        optional_split(
            dataset,
            name="drift",
            kind="ood_drift",
            query_file="test_drift.bin",
            neighbor_file="neighbors_drift.bin",
        ),
        optional_split(
            dataset,
            name="filter",
            kind="filtered",
            query_file="test_filter.bin",
            neighbor_file="neighbors_filter.bin",
            label_file="test_filter_topics.bin",
        ),
    ):
        if split is not None:
            rows.append(split)
    difficulty = dataset / "test_difficulty.bin"
    if difficulty.exists():
        labels = label_count(difficulty)
        if labels != test_shape[0]:
            raise SystemExit(
                f"test_difficulty.bin label count {labels} does not match "
                f"{test_shape[0]}"
            )
        rows[0]["difficulty_labels"] = "test_difficulty.bin"
    return rows


def vector_norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis=1)


def duplicate_fraction(vectors: np.ndarray) -> float:
    if len(vectors) == 0:
        return 0.0
    contiguous = np.ascontiguousarray(vectors)
    row_dtype = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    rows = contiguous.view(row_dtype).reshape(-1)
    return float(1.0 - np.unique(rows).size / len(rows))


def coordinate_dispersion(vectors: np.ndarray) -> dict[str, Any]:
    """Summarize per-coordinate spread without a full covariance eigensolve."""
    if vectors.size == 0:
        return {}
    values = np.asarray(vectors, dtype=np.float64)
    dim = values.shape[1]
    means = np.mean(values, axis=0)
    variances = np.var(values, axis=0)
    stds = np.sqrt(variances)
    total_variance = float(np.sum(variances))
    effective_dim = 0.0
    if total_variance > 0.0:
        weights = variances / total_variance
        effective_dim = float(1.0 / np.sum(weights * weights))
    return {
        "centroid_norm": float(np.linalg.norm(means)),
        "mean_abs_coordinate_mean": float(np.mean(np.abs(means))),
        "dimension_std": quantiles(stds),
        "dimension_variance": quantiles(variances),
        "total_variance": total_variance,
        "variance_effective_dim": effective_dim,
        "variance_effective_dim_fraction": float(effective_dim / dim) if dim else 0.0,
        "zero_variance_fraction": float(np.mean(variances <= 1e-12)),
    }


def pair_distances(
    train: np.ndarray,
    metric: Metric,
    pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(train) < 2 or pairs == 0:
        return np.array([], dtype=np.float32)
    left = rng.integers(0, len(train), size=pairs)
    right = rng.integers(0, len(train), size=pairs)
    same = left == right
    while np.any(same):
        right[same] = rng.integers(0, len(train), size=int(np.sum(same)))
        same = left == right
    a = train[left]
    b = train[right]
    if metric == "cosine":
        return (1.0 - np.sum(a * b, axis=1)).astype(np.float32)
    return np.linalg.norm(a - b, axis=1).astype(np.float32)


def distances_to_ids(
    query: np.ndarray,
    train: np.ndarray,
    ids: np.ndarray,
    metric: Metric,
) -> np.ndarray:
    valid = ids[ids >= 0]
    if len(valid) == 0:
        return np.array([], dtype=np.float32)
    vectors = train[valid]
    if metric == "cosine":
        return (1.0 - vectors @ query).astype(np.float32)
    return np.linalg.norm(vectors - query, axis=1).astype(np.float32)


def lid_mle_from_distances(distances: np.ndarray, lid_k: int) -> float:
    dists = np.sort(distances[np.isfinite(distances)])
    dists = dists[dists > 1e-12]
    k = min(lid_k, len(dists))
    if k < 3:
        return math.nan
    selected = dists[:k]
    r_max = float(selected[-1])
    if r_max <= 0.0:
        return math.nan
    ratios = np.clip(selected[:-1] / r_max, 1e-12, 1.0 - 1e-12)
    denom = float(np.sum(np.log(ratios)))
    if denom >= 0.0:
        return math.nan
    return float(-(k - 1) / denom)


def query_neighbor_metrics(
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
    query_ids: np.ndarray,
    metric: Metric,
    lid_k: int,
) -> dict[str, Any]:
    top1: list[float] = []
    top2_gap: list[float] = []
    top2_ratio: list[float] = []
    topk_spread: list[float] = []
    lids: list[float] = []
    for query_id in query_ids:
        dists = distances_to_ids(
            np.asarray(test[query_id], dtype=np.float32),
            train,
            np.asarray(neighbors[query_id]),
            metric,
        )
        if len(dists) == 0:
            continue
        dists = np.sort(dists)
        top1.append(float(dists[0]))
        if len(dists) >= 2:
            gap = float(dists[1] - dists[0])
            top2_gap.append(gap)
            ratio = float(dists[1] / max(dists[0], 1e-12))
            top2_ratio.append(ratio)
        if len(dists) >= 3:
            topk_spread.append(float(dists[-1] - dists[0]))
        lids.append(lid_mle_from_distances(dists, lid_k))
    return {
        "nearest_distance": quantiles(np.asarray(top1, dtype=np.float32)),
        "top2_gap": quantiles(np.asarray(top2_gap, dtype=np.float32)),
        "top2_ratio": quantiles(np.asarray(top2_ratio, dtype=np.float32)),
        "topk_spread": quantiles(np.asarray(topk_spread, dtype=np.float32)),
        "lid_mle": quantiles(np.asarray(lids, dtype=np.float32)),
    }


def sampled_relative_contrast(
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
    query_ids: np.ndarray,
    train_ids: np.ndarray,
    metric: Metric,
) -> dict[str, float]:
    sampled_train = np.asarray(train[train_ids], dtype=np.float32)
    contrasts: list[float] = []
    for query_id in query_ids:
        query = np.asarray(test[query_id], dtype=np.float32)
        true_nearest = distances_to_ids(
            query,
            train,
            np.asarray(neighbors[query_id, :1]),
            metric,
        )
        if len(true_nearest) == 0 or true_nearest[0] <= 1e-12:
            continue
        if metric == "cosine":
            sampled = 1.0 - sampled_train @ query
        else:
            sampled = np.linalg.norm(sampled_train - query, axis=1)
        contrasts.append(float(np.mean(sampled) / true_nearest[0]))
    return quantiles(np.asarray(contrasts, dtype=np.float32))


def hubness(neighbors: np.ndarray, train_count: int) -> dict[str, float]:
    top = np.asarray(neighbors[:, : min(10, neighbors.shape[1])])
    top = top[top >= 0]
    if len(top) == 0:
        return {}
    counts = np.bincount(top, minlength=train_count).astype(np.float64)
    nonzero = counts[counts > 0.0]
    mean = float(np.mean(counts))
    std = float(np.std(counts))
    skew = 0.0
    if std > 0.0:
        skew = float(np.mean((counts - mean) ** 3) / (std**3))
    sorted_counts = np.sort(counts)
    cumulative = np.cumsum(sorted_counts)
    gini = 0.0
    if cumulative[-1] > 0.0:
        n = len(sorted_counts)
        gini = float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)
    return {
        "topk": float(min(10, neighbors.shape[1])),
        "nonzero_fraction": float(len(nonzero) / train_count),
        "max_count": float(np.max(counts)),
        "p99_count": float(np.percentile(counts, 99)),
        "skewness": skew,
        "gini": gini,
    }


def gini(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    sorted_values = np.sort(values.astype(np.float64))
    total = float(np.sum(sorted_values))
    if total == 0.0:
        return 0.0
    cumulative = np.cumsum(sorted_values)
    n = len(sorted_values)
    return float((n + 1 - 2 * np.sum(cumulative) / total) / n)


def coarse_partition_imbalance(
    vectors: np.ndarray,
    metric: Metric,
    clusters: int,
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    if clusters <= 0 or len(vectors) == 0:
        return {}
    k = min(clusters, len(vectors))
    centroid_ids = rng.choice(len(vectors), size=k, replace=False)
    centroids = np.asarray(vectors[centroid_ids], dtype=np.float32).copy()
    assignments = np.zeros(len(vectors), dtype=np.int32)

    for _ in range(max(1, iterations)):
        if metric == "cosine":
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            centroids = centroids / np.maximum(norms, 1e-12)
            assignments = np.argmax(vectors @ centroids.T, axis=1).astype(np.int32)
        else:
            diff = vectors[:, None, :] - centroids[None, :, :]
            assignments = np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(
                np.int32
            )

        for cluster_id in range(k):
            members = vectors[assignments == cluster_id]
            if len(members) == 0:
                centroids[cluster_id] = vectors[rng.integers(0, len(vectors))]
            else:
                centroids[cluster_id] = np.mean(members, axis=0)

    counts = np.bincount(assignments, minlength=k).astype(np.float64)
    nonzero = counts[counts > 0.0]
    return {
        "clusters": float(k),
        "iterations": float(max(1, iterations)),
        "empty_fraction": float(1.0 - len(nonzero) / k),
        "count_p50": float(np.percentile(counts, 50)),
        "count_p95": float(np.percentile(counts, 95)),
        "count_max": float(np.max(counts)),
        "count_gini": gini(counts),
    }


def profile_dataset(
    dataset: Path,
    *,
    metric: Metric,
    sample_train: int,
    sample_queries: int,
    pair_samples: int,
    coarse_clusters: int,
    coarse_iters: int,
    lid_k: int,
    seed: int,
) -> dict[str, Any]:
    train = read_vec1(dataset / "train.bin")
    test = read_vec1(dataset / "test.bin")
    neighbors = read_nbr1(dataset / "neighbors.bin")
    if test.shape[0] != neighbors.shape[0]:
        message = (
            f"test query count {test.shape[0]} does not match "
            f"neighbors {neighbors.shape[0]}"
        )
        raise SystemExit(message)
    rng = np.random.default_rng(seed)
    train_ids = sample_indices(train.shape[0], sample_train, rng)
    query_ids = sample_indices(test.shape[0], sample_queries, rng)
    sampled_train = np.asarray(train[train_ids], dtype=np.float32)
    sampled_test = np.asarray(test[query_ids], dtype=np.float32)

    return {
        "dataset": str(dataset),
        "metric": metric,
        "shape": {
            "train": int(train.shape[0]),
            "queries": int(test.shape[0]),
            "dim": int(train.shape[1]),
            "ground_truth_k": int(neighbors.shape[1]),
        },
        "query_splits": query_splits(
            dataset,
            (int(test.shape[0]), int(test.shape[1])),
            int(neighbors.shape[1]),
        ),
        "samples": {
            "train": int(len(train_ids)),
            "queries": int(len(query_ids)),
            "pairs": int(
                min(pair_samples, max(0, len(train_ids) * (len(train_ids) - 1)))
            ),
            "seed": int(seed),
        },
        "norms": {
            "train": quantiles(vector_norms(sampled_train)),
            "queries": quantiles(vector_norms(sampled_test)),
        },
        "coordinate_dispersion_sample": coordinate_dispersion(sampled_train),
        "exact_duplicate_fraction_sample": duplicate_fraction(sampled_train),
        "pair_distance_sample": quantiles(
            pair_distances(sampled_train, metric, pair_samples, rng)
        ),
        "query_neighbors": query_neighbor_metrics(
            train,
            test,
            neighbors,
            query_ids,
            metric,
            lid_k,
        ),
        "sampled_relative_contrast": sampled_relative_contrast(
            train,
            test,
            neighbors,
            query_ids,
            train_ids,
            metric,
        ),
        "hubness": hubness(neighbors, train.shape[0]),
        "coarse_partition_imbalance": coarse_partition_imbalance(
            sampled_train,
            metric,
            coarse_clusters,
            coarse_iters,
            rng,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metric", choices=("auto", "cosine", "l2"), default="auto")
    parser.add_argument("--sample-train", type=int, default=4096)
    parser.add_argument("--sample-queries", type=int, default=1000)
    parser.add_argument("--pair-samples", type=int, default=20000)
    parser.add_argument("--coarse-clusters", type=int, default=64)
    parser.add_argument("--coarse-iters", type=int, default=8)
    parser.add_argument("--lid-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric = infer_metric(args.dataset, args.metric)
    result = profile_dataset(
        args.dataset,
        metric=metric,
        sample_train=args.sample_train,
        sample_queries=args.sample_queries,
        pair_samples=args.pair_samples,
        coarse_clusters=args.coarse_clusters,
        coarse_iters=args.coarse_iters,
        lid_k=args.lid_k,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        tmp = args.output.with_suffix(args.output.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(args.output)


if __name__ == "__main__":
    main()
