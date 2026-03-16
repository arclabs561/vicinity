#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "h5py", "requests"]
# ///
"""Download and convert ann-benchmarks datasets to vicinity's binary format.

Supports the standard ann-benchmarks.com datasets (HDF5).

Usage:
    uv run scripts/download_ann_benchmarks.py sift-128-euclidean
    uv run scripts/download_ann_benchmarks.py glove-25-angular
    uv run scripts/download_ann_benchmarks.py --list

Output: data/ann-benchmarks/<name>/{train,test,neighbors}.bin (VEC1/NBR1 format)

These files can be used directly with:
    JIN_DATASET=ann cargo run --example 03_quick_benchmark --release
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
import requests

# Standard ann-benchmarks datasets
# Format: (name, url_suffix, distance_metric, normalize)
DATASETS = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,  # Keep original metric; use original ground truth
        "size_mb": 501,
    },
    "glove-25-angular": {
        "url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "metric": "angular",
        "normalize": True,  # Normalize for cosine-based HNSW
        "size_mb": 121,
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "size_mb": 512,
    },
    "fashion-mnist-784-euclidean": {
        "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 217,
    },
    "nytimes-256-angular": {
        "url": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "size_mb": 312,
    },
    "mnist-784-euclidean": {
        "url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 217,
    },
}


def download_file(url: str, dest: Path) -> None:
    """Download with progress indicator."""
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {downloaded // (1024*1024)}MB / {total // (1024*1024)}MB ({pct:.0f}%)", end="", flush=True)
    print()


def write_vec1(path: Path, vectors: np.ndarray) -> None:
    """Write vectors in VEC1 binary format."""
    n, d = vectors.shape
    with open(path, "wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<II", n, d))
        f.write(vectors.astype(np.float32).tobytes())
    print(f"  Wrote {path}: {n} vectors x {d} dims")


def write_nbr1(path: Path, neighbors: np.ndarray) -> None:
    """Write neighbors in NBR1 binary format."""
    n, k = neighbors.shape
    with open(path, "wb") as f:
        f.write(b"NBR1")
        f.write(struct.pack("<II", n, k))
        f.write(neighbors.astype(np.int32).tobytes())
    print(f"  Wrote {path}: {n} queries x {k} neighbors")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize vectors (required for vicinity's HNSW cosine distance)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vectors / norms


def recompute_ground_truth(train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    """Recompute ground truth using cosine distance on normalized vectors.

    Processes queries in batches to avoid O(n_test * n_train) memory spike.
    """
    print(f"  Recomputing ground truth (cosine distance, k={k})...")
    n_test = test.shape[0]
    batch_size = max(1, min(100, 500_000_000 // (train.shape[0] * 4)))  # ~500MB per batch
    neighbors = np.empty((n_test, k), dtype=np.int32)

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        # Cosine similarity = dot(a, b) for normalized vectors
        sims = test[start:end] @ train.T  # (batch, n_train)
        # Top-k by similarity (highest = closest)
        # Use argpartition for O(n) partial sort instead of O(n log n) full sort
        if k < sims.shape[1]:
            top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
            # Sort the top-k by similarity (descending)
            for i in range(end - start):
                order = np.argsort(-sims[i, top_k_idx[i]])
                top_k_idx[i] = top_k_idx[i, order]
            neighbors[start:end] = top_k_idx
        else:
            neighbors[start:end] = np.argsort(-sims, axis=1)[:, :k]

        if end < n_test:
            print(f"    {end}/{n_test} queries...", flush=True)

    return neighbors


def convert_dataset(name: str, info: dict, output_dir: Path) -> None:
    """Download HDF5, convert to VEC1/NBR1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_path = output_dir / f"{name}.hdf5"
    if not hdf5_path.exists():
        download_file(info["url"], hdf5_path)
    else:
        print(f"Using cached {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        gt_neighbors = np.array(f["neighbors"])

    print(f"  Train: {train.shape}, Test: {test.shape}, GT: {gt_neighbors.shape}")

    # Normalize for cosine-based HNSW
    if info.get("normalize", False):
        print("  Normalizing vectors (L2)...")
        train = normalize_vectors(train)
        test = normalize_vectors(test)
        # Recompute ground truth on normalized vectors using cosine distance
        k = gt_neighbors.shape[1]
        gt_neighbors = recompute_ground_truth(train, test, k)

    write_vec1(output_dir / "train.bin", train)
    write_vec1(output_dir / "test.bin", test)
    write_nbr1(output_dir / "neighbors.bin", gt_neighbors)

    print(f"\nDataset ready: {output_dir}/")
    print(f"Run benchmark:")
    print(f"  cargo run --example 04_rigorous_benchmark --release -- --data-dir {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ann-benchmarks datasets")
    parser.add_argument("dataset", nargs="?", help="Dataset name (e.g., sift-128-euclidean)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--output", default="data/ann-benchmarks", help="Output directory")
    args = parser.parse_args()

    if args.list or not args.dataset:
        print("Available datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name:<35} {info['metric']:<12} ~{info['size_mb']}MB")
        print(f"\nUsage: uv run {sys.argv[0]} <dataset-name>")
        return

    if args.dataset not in DATASETS:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    output_dir = Path(args.output) / args.dataset
    convert_dataset(args.dataset, DATASETS[args.dataset], output_dir)


if __name__ == "__main__":
    main()
