#!/usr/bin/env python3
"""Generate a tiny deterministic VEC1/NBR1 dataset for benchmark smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("zero vector cannot be normalized")
    return [value / norm for value in vector]


def make_train(n: int, dim: int) -> list[list[float]]:
    rng = random.Random(42)
    vectors: list[list[float]] = []
    for i in range(n):
        base = [0.0] * dim
        base[i % dim] = 1.0
        base[(i * 3 + 1) % dim] += 0.35
        base[(i * 5 + 2) % dim] += 0.15
        jitter = [(rng.random() - 0.5) * 0.03 for _ in range(dim)]
        vectors.append(normalize([base[j] + jitter[j] for j in range(dim)]))
    return vectors


def make_test(train: list[list[float]], n: int) -> list[list[float]]:
    dim = len(train[0])
    queries: list[list[float]] = []
    for i in range(n):
        a = train[(i * 7) % len(train)]
        b = train[(i * 11 + 3) % len(train)]
        queries.append(normalize([0.92 * a[j] + 0.08 * b[j] for j in range(dim)]))
    return queries


def cosine(a: list[float], b: list[float]) -> float:
    return sum(a[j] * b[j] for j in range(len(a)))


def ground_truth(
    train: list[list[float]],
    test: list[list[float]],
    k: int,
) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for query in test:
        scored = sorted(
            ((cosine(query, vector), idx) for idx, vector in enumerate(train)),
            reverse=True,
        )
        neighbors.append([idx for _, idx in scored[:k]])
    return neighbors


def write_vec1(path: Path, vectors: list[list[float]]) -> None:
    n = len(vectors)
    dim = len(vectors[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<II", n, dim))
        for vector in vectors:
            f.write(struct.pack(f"<{dim}f", *vector))
    tmp.replace(path)


def write_nbr1(path: Path, neighbors: list[list[int]]) -> None:
    n = len(neighbors)
    k = len(neighbors[0])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(b"NBR1")
        f.write(struct.pack("<II", n, k))
        for row in neighbors:
            f.write(struct.pack(f"<{k}i", *row))
    tmp.replace(path)


def binary_shape(path: Path, magic: bytes) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except FileNotFoundError:
        return None
    if len(header) != 12 or header[:4] != magic:
        return None
    rows, width = struct.unpack("<II", header[4:])
    expected_size = 12 + rows * width * 4
    if path.stat().st_size != expected_size:
        return None
    return rows, width


def read_vec1_shape(path: Path) -> tuple[int, int] | None:
    return binary_shape(path, b"VEC1")


def read_nbr1_shape(path: Path) -> tuple[int, int] | None:
    return binary_shape(path, b"NBR1")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_manifest(train: int, test: int, dim: int, k: int) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "train": train,
        "test": test,
        "dim": dim,
        "k": k,
    }


def manifest_with_outputs(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    full_manifest = dict(manifest)
    full_manifest["outputs"] = {
        filename: {
            "bytes": (output / filename).stat().st_size,
            "sha256": file_sha256(output / filename),
        }
        for filename in ("train.bin", "test.bin", "neighbors.bin")
    }
    return full_manifest


def matching_outputs(output: Path, manifest: dict[str, Any]) -> bool:
    manifest_path = output / "ann_smoke_dataset.json"
    try:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if any(existing_manifest.get(key) != value for key, value in manifest.items()):
        return False
    if read_vec1_shape(output / "train.bin") != (manifest["train"], manifest["dim"]):
        return False
    if read_vec1_shape(output / "test.bin") != (manifest["test"], manifest["dim"]):
        return False
    if read_nbr1_shape(output / "neighbors.bin") != (manifest["test"], manifest["k"]):
        return False
    try:
        return existing_manifest == manifest_with_outputs(output, manifest)
    except FileNotFoundError:
        return False


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k > args.train:
        raise SystemExit("--k must be <= --train")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = expected_manifest(args.train, args.test, args.dim, args.k)
    if not args.force and matching_outputs(args.output, manifest):
        print(f"Reusing ANN smoke dataset at {args.output}")
        return

    train = make_train(args.train, args.dim)
    test = make_test(train, args.test)
    neighbors = ground_truth(train, test, args.k)

    write_vec1(args.output / "train.bin", train)
    write_vec1(args.output / "test.bin", test)
    write_nbr1(args.output / "neighbors.bin", neighbors)
    write_manifest_atomic(
        args.output / "ann_smoke_dataset.json",
        manifest_with_outputs(args.output, manifest),
    )
    print(f"Wrote ANN smoke dataset to {args.output}")


if __name__ == "__main__":
    main()
