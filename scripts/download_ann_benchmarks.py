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

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Standard ann-benchmarks datasets
# Format: (name, url_suffix, distance_metric, normalize)
DATASETS = {
    "sift-128-euclidean": {
        "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,  # Keep original metric; use original ground truth
        "size_mb": 501,
        "expected_bytes": 525_128_288,
        "expected_sha256": (
            "dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984"
        ),
    },
    "glove-25-angular": {
        "url": "https://ann-benchmarks.com/glove-25-angular.hdf5",
        "metric": "angular",
        "normalize": True,  # Normalize for cosine-based HNSW
        "recompute_ground_truth": True,
        "size_mb": 121,
        "expected_bytes": 127_359_688,
        "expected_sha256": (
            "51004cb0ae962159f0db507a51fec2b395de14b166f55976c89f16bd2f8b6391"
        ),
    },
    "glove-50-angular": {
        "url": "https://ann-benchmarks.com/glove-50-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 235,
        "expected_bytes": 246_711_088,
        "expected_sha256": (
            "388b0aedc2dad689549e6587932c8c9efeaf8a95f383f36dc567a40233a11f40"
        ),
    },
    "glove-100-angular": {
        "url": "https://ann-benchmarks.com/glove-100-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 463,
        "expected_bytes": 485_413_888,
        "expected_sha256": (
            "544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4"
        ),
    },
    "glove-200-angular": {
        "url": "https://ann-benchmarks.com/glove-200-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 918,
        "expected_bytes": 962_819_488,
        "expected_sha256": (
            "4839085e5a8bb293434a1a66e1aa0193afc3f07c6797a85f1dbd91656172da20"
        ),
    },
    "fashion-mnist-784-euclidean": {
        "url": "https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 217,
        "expected_bytes": 227_528_288,
        "expected_sha256": (
            "7f24e5122b71346346686a6ea80e08d9145b9c020f9f2db4c4c799dfcfc96a07"
        ),
    },
    "nytimes-256-angular": {
        "url": "https://ann-benchmarks.com/nytimes-256-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "size_mb": 301,
        "expected_bytes": 315_208_288,
        "expected_sha256": (
            "6ad5d2cde6d6a6f21528d541901304ebfd40885d6ce6afae37bccbe278254a93"
        ),
    },
    "mnist-784-euclidean": {
        "url": "https://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 217,
        "expected_bytes": 227_528_288,
        "expected_sha256": (
            "35ff29594d96d1c51691bdef900f6adc10e33bac099ac1f37330ed5fcf652336"
        ),
    },
    "gist-960-euclidean": {
        "url": "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "metric": "euclidean",
        "normalize": False,
        "size_mb": 3600,
        "expected_bytes": 3_844_648_288,
        "expected_sha256": (
            "8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a"
        ),
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
        "expected_sha256": (
            "a0a44dfe80c58e63860862eeea2d34e62cff958dc3aec14fd65263d1d3c751f8"
        ),
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


def verify_expected_sha256(path: Path, expected_sha256: str) -> None:
    """Fail when a cached or downloaded HDF5 does not match the pinned hash."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise SystemExit(
            f"{path} has SHA256 {actual_sha256}, expected {expected_sha256}. "
            "Re-run with --redownload to replace it."
        )


def require_numpy():
    """Import NumPy only for conversion paths."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dataset conversion requires numpy. Run this script with "
            "`uv run scripts/download_ann_benchmarks.py ...`."
        ) from exc
    return np


def require_h5py():
    """Import h5py only for conversion paths."""
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dataset conversion requires h5py. Run this script with "
            "`uv run scripts/download_ann_benchmarks.py ...`."
        ) from exc
    return h5py


def download_file(
    url: str,
    dest: Path,
    *,
    redownload: bool = False,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> None:
    """Download with progress indicator, writing atomically on success."""
    if dest.exists() and not redownload:
        if expected_bytes is not None:
            verify_expected_size(dest, expected_bytes)
        if expected_sha256 is not None:
            verify_expected_sha256(dest, expected_sha256)
        print(f"Using cached {dest}")
        return

    import requests

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
    if expected_sha256 is not None:
        verify_expected_sha256(tmp, expected_sha256)
    tmp.replace(dest)
    print()


def write_vec1(path: Path, vectors: np.ndarray) -> None:
    """Write vectors in VEC1 binary format, atomically on success."""
    np = require_numpy()
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
    np = require_numpy()
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
    np = require_numpy()
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return vectors / norms


def recompute_ground_truth(train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    """Recompute ground truth using cosine distance on normalized vectors.

    Processes queries in batches to avoid O(n_test * n_train) memory spike.
    """
    np = require_numpy()
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


def binary_output_shape(path: Path, magic: bytes) -> tuple[int, int] | None:
    """Return the binary output shape when magic and byte length are valid."""
    if not path.exists():
        return None
    with path.open("rb") as f:
        header = f.read(12)
    if len(header) != 12 or header[:4] != magic:
        return None

    rows, width = struct.unpack("<II", header[4:])
    expected_size = 12 + rows * width * 4
    if path.stat().st_size != expected_size:
        return None
    return rows, width


def valid_binary_output(path: Path, magic: bytes) -> bool:
    """Return true when an output has the expected magic and byte length."""
    return binary_output_shape(path, magic) is not None


def existing_output_shapes(output_dir: Path) -> dict[str, list[int]]:
    """Return shapes for already-converted binary outputs."""
    outputs = {
        "train_shape": binary_output_shape(output_dir / "train.bin", b"VEC1"),
        "test_shape": binary_output_shape(output_dir / "test.bin", b"VEC1"),
        "neighbors_shape": binary_output_shape(output_dir / "neighbors.bin", b"NBR1"),
    }
    missing = [name for name, shape in outputs.items() if shape is None]
    if missing:
        raise SystemExit(
            "Existing converted outputs are missing or invalid: " + ", ".join(missing)
        )
    return {name: list(shape) for name, shape in outputs.items() if shape is not None}


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
        "expected_sha256": info.get("expected_sha256"),
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


def write_complete_manifest_from_shapes(
    output_dir: Path,
    name: str,
    info: dict,
    hdf5_path: Path,
    output_shapes: dict[str, list[int]],
) -> None:
    """Write conversion metadata after binary outputs are in place."""
    manifest = {
        "complete": True,
        "settings": conversion_settings(name, info),
        "hdf5": {
            "path": str(hdf5_path.name),
            "bytes": hdf5_path.stat().st_size,
            "sha256": sha256_file(hdf5_path),
        },
        "outputs": output_shapes,
    }
    write_manifest_atomically(output_dir, manifest)


def write_complete_manifest(
    output_dir: Path,
    name: str,
    info: dict,
    hdf5_path: Path,
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
) -> None:
    """Write conversion metadata after freshly converted outputs are in place."""
    write_complete_manifest_from_shapes(
        output_dir,
        name,
        info,
        hdf5_path,
        {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "neighbors_shape": list(neighbors.shape),
        },
    )


def adopt_existing_outputs(
    output_dir: Path,
    name: str,
    info: dict,
    hdf5_path: Path,
) -> None:
    """Trust already-converted outputs and write a current manifest."""
    if not hdf5_path.exists():
        raise SystemExit(
            f"Cannot adopt existing outputs: cached HDF5 is missing at {hdf5_path}."
        )
    if info.get("expected_bytes") is not None:
        verify_expected_size(hdf5_path, info["expected_bytes"])
    if info.get("expected_sha256") is not None:
        verify_expected_sha256(hdf5_path, info["expected_sha256"])

    write_complete_manifest_from_shapes(
        output_dir,
        name,
        info,
        hdf5_path,
        existing_output_shapes(output_dir),
    )


def validate_hdf5_arrays(
    train: np.ndarray, test: np.ndarray, neighbors: np.ndarray
) -> None:
    """Validate basic ann-benchmarks shape and index invariants."""
    np = require_numpy()
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
    adopt_existing: bool = False,
) -> None:
    """Download HDF5, convert to VEC1/NBR1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / f"{name}.hdf5"

    if (
        converted_outputs_exist(output_dir)
        and manifest_matches(output_dir, name, info)
        and not force
        and not redownload
    ):
        print(f"Dataset already converted: {output_dir}/")
        print("Use --force to rebuild train.bin/test.bin/neighbors.bin.")
        return

    if converted_outputs_exist(output_dir) and not force and not redownload:
        if adopt_existing:
            adopt_existing_outputs(output_dir, name, info, hdf5_path)
            print(f"Adopted existing converted dataset: {output_dir}/")
            return
        raise SystemExit(
            f"Converted outputs already exist in {output_dir}, but dataset.json "
            "is missing or does not match the current conversion settings. "
            "Use --force to rebuild, or --adopt-existing to trust the existing "
            "train.bin/test.bin/neighbors.bin files and write a manifest."
        )

    write_incomplete_manifest(output_dir, name, info)
    download_file(
        info["url"],
        hdf5_path,
        redownload=redownload,
        expected_bytes=info.get("expected_bytes"),
        expected_sha256=info.get("expected_sha256"),
    )

    try:
        h5py = require_h5py()
        np = require_numpy()

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


def download_dataset_hdf5(
    name: str,
    info: dict,
    output_dir: Path,
    *,
    redownload: bool,
) -> None:
    """Download and verify the source HDF5 without converting it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = output_dir / f"{name}.hdf5"
    download_file(
        info["url"],
        hdf5_path,
        redownload=redownload,
        expected_bytes=info.get("expected_bytes"),
        expected_sha256=info.get("expected_sha256"),
    )
    print(f"  SHA256: {sha256_file(hdf5_path)}")


def selected_dataset_names(dataset: str | None, all_datasets: bool) -> list[str]:
    """Return the datasets requested by CLI arguments."""
    if all_datasets:
        if dataset:
            raise SystemExit("--all cannot be combined with a dataset name.")
        return list(DATASETS)

    if not dataset:
        raise SystemExit("dataset is required unless --list or --all is used")

    if dataset not in DATASETS:
        raise SystemExit(
            f"Unknown dataset: {dataset}\nAvailable: {', '.join(DATASETS.keys())}"
        )

    return [dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ann-benchmarks datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name (e.g., sift-128-euclidean)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every configured dataset",
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
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and verify the source HDF5 without converting it",
    )
    parser.add_argument(
        "--adopt-existing",
        action="store_true",
        help=(
            "Write dataset.json for existing converted .bin files after "
            "validating their headers and cached HDF5 size"
        ),
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:\n")
        for name, info in DATASETS.items():
            print(f"  {name:<35} {info['metric']:<12} ~{info['size_mb']}MB")
        print(f"\nUsage: uv run {sys.argv[0]} <dataset-name>")
        return

    for dataset in selected_dataset_names(args.dataset, args.all):
        output_dir = Path(args.output) / dataset
        if args.download_only:
            download_dataset_hdf5(
                dataset,
                DATASETS[dataset],
                output_dir,
                redownload=args.redownload,
            )
            continue

        convert_dataset(
            dataset,
            DATASETS[dataset],
            output_dir,
            force=args.force,
            redownload=args.redownload,
            adopt_existing=args.adopt_existing,
        )


if __name__ == "__main__":
    main()
