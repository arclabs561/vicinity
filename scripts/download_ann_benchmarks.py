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
    uv run scripts/download_ann_benchmarks.py glove-25-angular --force
    uv run scripts/download_ann_benchmarks.py --list

Output: data/ann-benchmarks/<name>/{train,test,neighbors}.bin (VEC1/NBR1 format)

These files can be used directly with:
    cargo run --example ann_benchmark --release --features hnsw -- \
        data/ann-benchmarks/<name>
"""

import argparse
import hashlib
import json
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
        "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,  # Keep original metric; use original ground truth
        "size_mb": 501,
        "expected_bytes": 525_128_288,
    },
    "glove-25-angular": {
        "url": "https://ann-benchmarks.com/glove-25-angular.hdf5",
        "metric": "angular",
        "normalize": True,  # Normalize for cosine-based HNSW
        "recompute_ground_truth": True,
        "size_mb": 121,
        "expected_bytes": 127_359_688,
    },
    "glove-50-angular": {
        "url": "https://ann-benchmarks.com/glove-50-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 235,
        "expected_bytes": 246_711_088,
    },
    "glove-100-angular": {
        "url": "https://ann-benchmarks.com/glove-100-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 463,
        "expected_bytes": 485_413_888,
    },
    "glove-200-angular": {
        "url": "https://ann-benchmarks.com/glove-200-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 918,
        "expected_bytes": 962_819_488,
    },
    "fashion-mnist-784-euclidean": {
        "url": "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 217,
        "expected_bytes": 227_528_288,
    },
    "nytimes-256-angular": {
        "url": "https://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 301,
        "expected_bytes": 315_208_288,
    },
    "mnist-784-euclidean": {
        "url": "https://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 217,
        "expected_bytes": 227_528_288,
    },
    "gist-960-euclidean": {
        "url": "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 3600,
        "expected_bytes": 3_844_648_288,
    },
    "deep-image-96-angular": {
        "url": "https://ann-benchmarks.com/deep-image-96-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        # The HDF5 file already includes angular ground truth. Recomputing
        # against 9.99M training vectors is too expensive for a converter.
        "recompute_ground_truth": False,
        "size_mb": 3669,
        "expected_bytes": 3_848_008_288,
    },
}


def verify_expected_size(path: Path, expected_bytes: int) -> None:
    """Fail when a cached or downloaded HDF5 does not match the expected size."""
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"{path} has {actual_bytes} bytes, expected {expected_bytes}. "
            "Re-run with --redownload to replace it."
        )


def sha256_file(path: Path) -> str:
    """Hash a local file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    redownload: bool = False,
    expected_bytes: int | None = None,
) -> None:
    """Download with progress indicator, writing atomically on success."""
    if dest.exists() and not redownload:
        if expected_bytes is not None:
            verify_expected_size(dest, expected_bytes)
        print(f"Using cached {dest}")
        return

    print(f"Downloading {url} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(
                        f"\r  {downloaded // (1024 * 1024)}MB / "
                        f"{total // (1024 * 1024)}MB ({pct:.0f}%)",
                        end="",
                        flush=True,
                    )
    if expected_bytes is not None:
        verify_expected_size(tmp, expected_bytes)
    tmp.replace(dest)
    print()


def write_vec1(path: Path, vectors: np.ndarray) -> None:
    """Write vectors in VEC1 binary format, atomically on success."""
    n, d = vectors.shape
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<II", n, d))
        f.write(vectors.astype(np.float32).tobytes())
    tmp.replace(path)
    print(f"  Wrote {path}: {n} vectors x {d} dims")


def write_nbr1(path: Path, neighbors: np.ndarray) -> None:
    """Write neighbors in NBR1 binary format, atomically on success."""
    n, k = neighbors.shape
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"NBR1")
        f.write(struct.pack("<II", n, k))
        f.write(neighbors.astype(np.int32).tobytes())
    tmp.replace(path)
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
    # Keep each similarity batch near 500MB.
    batch_size = max(1, min(100, 500_000_000 // (train.shape[0] * 4)))
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


def valid_binary_output(path: Path, magic: bytes) -> bool:
    """Return true when an output has the expected magic and byte length."""
    if not path.exists():
        return False
    with path.open("rb") as f:
        header = f.read(12)
    if len(header) != 12 or header[:4] != magic:
        return False

    rows, width = struct.unpack("<II", header[4:])
    expected_size = 12 + rows * width * 4
    return path.stat().st_size == expected_size


def converted_outputs_exist(output_dir: Path) -> bool:
    """Return true when the converted dataset files are already present."""
    return (
        valid_binary_output(output_dir / "train.bin", b"VEC1")
        and valid_binary_output(output_dir / "test.bin", b"VEC1")
        and valid_binary_output(output_dir / "neighbors.bin", b"NBR1")
    )


def manifest_path(output_dir: Path) -> Path:
    """Return the manifest path for a converted dataset."""
    return output_dir / "dataset.json"


def conversion_settings(name: str, info: dict) -> dict:
    """Return the conversion settings that must match for idempotent reuse."""
    return {
        "version": 1,
        "dataset": name,
        "url": info["url"],
        "metric": info["metric"],
        "normalize": info.get("normalize", False),
        "recompute_ground_truth": info.get("recompute_ground_truth", False),
        "expected_bytes": info.get("expected_bytes"),
    }


def manifest_matches(output_dir: Path, name: str, info: dict) -> bool:
    """Return true when the output manifest matches the current conversion config."""
    path = manifest_path(output_dir)
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return manifest.get("complete") is True and manifest.get(
        "settings"
    ) == conversion_settings(name, info)


def write_manifest_atomically(output_dir: Path, manifest: dict) -> None:
    """Write the dataset manifest atomically."""
    path = manifest_path(output_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_incomplete_manifest(output_dir: Path, name: str, info: dict) -> None:
    """Mark conversion as in-progress before replacing dataset outputs."""
    write_manifest_atomically(
        output_dir,
        {
            "complete": False,
            "settings": conversion_settings(name, info),
        },
    )


def write_complete_manifest(
    output_dir: Path,
    name: str,
    info: dict,
    hdf5_path: Path,
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
) -> None:
    """Write conversion metadata after all binary outputs are in place."""
    manifest = {
        "complete": True,
        "settings": conversion_settings(name, info),
        "hdf5": {
            "path": str(hdf5_path.name),
            "bytes": hdf5_path.stat().st_size,
            "sha256": sha256_file(hdf5_path),
        },
        "outputs": {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "neighbors_shape": list(neighbors.shape),
        },
    }
    write_manifest_atomically(output_dir, manifest)


def validate_hdf5_arrays(
    train: np.ndarray, test: np.ndarray, neighbors: np.ndarray
) -> None:
    """Validate basic ann-benchmarks shape and index invariants."""
    if train.ndim != 2 or test.ndim != 2 or neighbors.ndim != 2:
        raise SystemExit("Expected train, test, and neighbors to be 2D arrays.")
    if train.shape[1] != test.shape[1]:
        raise SystemExit(
            f"Train/test dimensions differ: {train.shape[1]} vs {test.shape[1]}."
        )
    if neighbors.shape[0] != test.shape[0]:
        neighbor_rows = neighbors.shape[0]
        test_rows = test.shape[0]
        raise SystemExit(
            f"Neighbor rows ({neighbor_rows}) must match test rows ({test_rows})."
        )
    if neighbors.shape[1] == 0:
        raise SystemExit("Ground-truth neighbor matrix must have at least one column.")
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise SystemExit("Train/test vectors contain NaN or infinite values.")
    if np.any(neighbors < 0) or np.any(neighbors >= train.shape[0]):
        raise SystemExit(
            "Ground-truth neighbor IDs are outside the train vector range."
        )


def convert_dataset(
    name: str,
    info: dict,
    output_dir: Path,
    *,
    force: bool,
    redownload: bool,
) -> None:
    """Download HDF5, convert to VEC1/NBR1."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if (
        converted_outputs_exist(output_dir)
        and manifest_matches(output_dir, name, info)
        and not force
        and not redownload
    ):
        print(f"Dataset already converted: {output_dir}/")
        print("Use --force to rebuild train.bin/test.bin/neighbors.bin.")
        return

    hdf5_path = output_dir / f"{name}.hdf5"
    write_incomplete_manifest(output_dir, name, info)
    download_file(
        info["url"],
        hdf5_path,
        redownload=redownload,
        expected_bytes=info.get("expected_bytes"),
    )

    try:
        with h5py.File(hdf5_path, "r") as f:
            train = np.array(f["train"])
            test = np.array(f["test"])
            gt_neighbors = np.array(f["neighbors"])
    except OSError as exc:
        raise SystemExit(
            f"Could not read cached HDF5 at {hdf5_path}. "
            "Re-run with --redownload to replace it."
        ) from exc

    print(f"  Train: {train.shape}, Test: {test.shape}, GT: {gt_neighbors.shape}")
    validate_hdf5_arrays(train, test, gt_neighbors)

    # Normalize for cosine-based HNSW
    if info.get("normalize", False):
        print("  Normalizing vectors (L2)...")
        train = normalize_vectors(train)
        test = normalize_vectors(test)
        if info.get("recompute_ground_truth", False):
            # Recompute ground truth on normalized vectors using cosine distance.
            k = gt_neighbors.shape[1]
            gt_neighbors = recompute_ground_truth(train, test, k)

    write_vec1(output_dir / "train.bin", train)
    write_vec1(output_dir / "test.bin", test)
    write_nbr1(output_dir / "neighbors.bin", gt_neighbors)
    write_complete_manifest(
        output_dir, name, info, hdf5_path, train, test, gt_neighbors
    )

    print(f"\nDataset ready: {output_dir}/")
    print("Run benchmark:")
    print(
        f"  cargo run --example ann_benchmark --release --features hnsw -- {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ann-benchmarks datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g., sift-128-euclidean)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--output",
        default="data/ann-benchmarks",
        help="Output directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild converted .bin files",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Replace the cached HDF5 file",
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name:<35} {info['metric']:<12} ~{info['size_mb']}MB")
        print(f"\nUsage: uv run {sys.argv[0]} <dataset-name>")
        return

    if not args.dataset:
        parser.error("dataset is required unless --list is used")

    if args.dataset not in DATASETS:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    output_dir = Path(args.output) / args.dataset
    convert_dataset(
        args.dataset,
        DATASETS[args.dataset],
        output_dir,
        force=args.force,
        redownload=args.redownload,
    )


if __name__ == "__main__":
    main()
