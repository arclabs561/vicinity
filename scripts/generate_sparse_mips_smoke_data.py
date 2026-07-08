#!/usr/bin/env python3
"""Generate a deterministic sparse MIPS smoke dataset.

The dataset uses:
- SPV1 for sparse vectors: CSR-style offsets, indices, and f32 values.
- NBR1 for exact MIPS ground-truth neighbors.

This is a small synthetic harness for SparseMIPS correctness and benchmark
plumbing. It is not a substitute for SPLADE or BM25 benchmark data.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path
from typing import Any, TypeAlias

MANIFEST_VERSION = 1
SPV1_MAGIC = b"SPV1"
NBR1_MAGIC = b"NBR1"

SparseVector: TypeAlias = tuple[list[int], list[float]]


def compact_pairs(pairs: list[tuple[int, float]]) -> SparseVector:
    """Sort by dimension and sum duplicate entries."""
    pairs.sort(key=lambda pair: pair[0])
    indices: list[int] = []
    values: list[float] = []
    for idx, value in pairs:
        if indices and indices[-1] == idx:
            values[-1] += value
        else:
            indices.append(idx)
            values.append(value)
    return indices, values


def make_sparse_vector(
    rng: random.Random,
    *,
    topic: int,
    vocab: int,
    topics: int,
    nnz: int,
) -> SparseVector:
    """Build one sparse vector with a topic-heavy term distribution."""
    topic_span = max(nnz * 4, vocab // max(topics, 1))
    topic_start = (topic * vocab) // max(topics, 1)
    pairs: list[tuple[int, float]] = []

    for slot in range(nnz):
        if rng.random() < 0.75:
            idx = (topic_start + rng.randrange(topic_span)) % vocab
            value = 0.8 + rng.random() * 1.8
        else:
            idx = rng.randrange(vocab)
            value = 0.05 + rng.random() * 0.45
        pairs.append((idx, value))

        # Deliberately emit a few duplicate dimensions. Real sparse retrieval
        # pipelines often accumulate term weights from multiple sources before
        # compaction, and SparseVector::from_pairs should match that boundary.
        if slot % 11 == 0:
            pairs.append((idx, value * 0.25))

    return compact_pairs(pairs)


def make_vectors(
    n: int,
    *,
    vocab: int,
    topics: int,
    nnz: int,
    seed: int,
) -> list[SparseVector]:
    rng = random.Random(seed)
    return [
        make_sparse_vector(
            rng,
            topic=i % topics,
            vocab=vocab,
            topics=topics,
            nnz=nnz,
        )
        for i in range(n)
    ]


def make_queries(
    n: int,
    *,
    vocab: int,
    topics: int,
    nnz: int,
    seed: int,
) -> list[SparseVector]:
    rng = random.Random(seed)
    return [
        make_sparse_vector(
            rng,
            topic=(i * 7) % topics,
            vocab=vocab,
            topics=topics,
            nnz=nnz,
        )
        for i in range(n)
    ]


def sparse_dot(left: SparseVector, right: SparseVector) -> float:
    left_idx, left_val = left
    right_idx, right_val = right
    li = 0
    ri = 0
    total = 0.0
    while li < len(left_idx) and ri < len(right_idx):
        if left_idx[li] == right_idx[ri]:
            total += left_val[li] * right_val[ri]
            li += 1
            ri += 1
        elif left_idx[li] < right_idx[ri]:
            li += 1
        else:
            ri += 1
    return total


def ground_truth(
    train: list[SparseVector],
    queries: list[SparseVector],
    k: int,
) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for query in queries:
        scored = sorted(
            ((sparse_dot(query, vector), idx) for idx, vector in enumerate(train)),
            key=lambda item: (-item[0], item[1]),
        )
        neighbors.append([idx for _, idx in scored[:k]])
    return neighbors


def flatten_vectors(
    vectors: list[SparseVector],
) -> tuple[list[int], list[int], list[float]]:
    offsets = [0]
    indices: list[int] = []
    values: list[float] = []
    for vector_indices, vector_values in vectors:
        if len(vector_indices) != len(vector_values):
            raise ValueError("sparse vector indices and values differ in length")
        indices.extend(vector_indices)
        values.extend(vector_values)
        offsets.append(len(indices))
    return offsets, indices, values


def write_spv1(path: Path, vectors: list[SparseVector]) -> tuple[int, int]:
    rows = len(vectors)
    offsets, indices, values = flatten_vectors(vectors)
    total_nnz = len(indices)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(SPV1_MAGIC)
        f.write(struct.pack("<IQ", rows, total_nnz))
        for offset in offsets:
            f.write(struct.pack("<Q", offset))
        for idx in indices:
            f.write(struct.pack("<I", idx))
        for value in values:
            f.write(struct.pack("<f", value))
    tmp.replace(path)
    return rows, total_nnz


def write_nbr1(path: Path, neighbors: list[list[int]]) -> tuple[int, int]:
    rows = len(neighbors)
    k = len(neighbors[0]) if neighbors else 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(NBR1_MAGIC)
        f.write(struct.pack("<II", rows, k))
        for row in neighbors:
            if len(row) != k:
                raise ValueError("ground-truth rows have inconsistent widths")
            for neighbor in row:
                f.write(struct.pack("<i", neighbor))
    tmp.replace(path)
    return rows, k


def read_spv1_shape(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(16)
    except FileNotFoundError:
        return None
    if len(header) != 16 or header[:4] != SPV1_MAGIC:
        return None
    rows, total_nnz = struct.unpack("<IQ", header[4:])
    expected_size = 16 + (rows + 1) * 8 + total_nnz * 8
    if path.stat().st_size != expected_size:
        return None
    return rows, total_nnz


def read_nbr1_shape(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(12)
    except FileNotFoundError:
        return None
    if len(header) != 12 or header[:4] != NBR1_MAGIC:
        return None
    rows, k = struct.unpack("<II", header[4:])
    expected_size = 12 + rows * k * 4
    if path.stat().st_size != expected_size:
        return None
    return rows, k


def expected_settings(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": args.train,
        "test": args.test,
        "vocab": args.vocab,
        "topics": args.topics,
        "nnz": args.nnz,
        "k": args.k,
        "seed": args.seed,
    }


def manifest_for(
    settings: dict[str, int],
    *,
    train_shape: tuple[int, int],
    test_shape: tuple[int, int],
    neighbors_shape: tuple[int, int],
) -> dict[str, Any]:
    return {
        "version": MANIFEST_VERSION,
        "format": "SPV1+NBR1",
        "settings": settings,
        "outputs": {
            "train_shape": list(train_shape),
            "test_shape": list(test_shape),
            "neighbors_shape": list(neighbors_shape),
        },
    }


def matching_outputs(output: Path, settings: dict[str, int]) -> bool:
    manifest_path = output / "sparse_mips_dataset.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if manifest.get("version") != MANIFEST_VERSION:
        return False
    if manifest.get("settings") != settings:
        return False

    train_shape = read_spv1_shape(output / "train.spv1")
    test_shape = read_spv1_shape(output / "test.spv1")
    neighbors_shape = read_nbr1_shape(output / "neighbors.bin")
    if train_shape is None or test_shape is None or neighbors_shape is None:
        return False

    outputs = manifest.get("outputs", {})
    return (
        outputs.get("train_shape") == list(train_shape)
        and outputs.get("test_shape") == list(test_shape)
        and outputs.get("neighbors_shape") == list(neighbors_shape)
        and train_shape[0] == settings["train"]
        and test_shape[0] == settings["test"]
        and neighbors_shape == (settings["test"], settings["k"])
    )


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
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--test", type=int, default=40)
    parser.add_argument("--vocab", type=int, default=2048)
    parser.add_argument("--topics", type=int, default=16)
    parser.add_argument("--nnz", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train <= 0 or args.test <= 0:
        raise SystemExit("--train and --test must be positive")
    if args.vocab <= 0 or args.topics <= 0 or args.nnz <= 0:
        raise SystemExit("--vocab, --topics, and --nnz must be positive")
    if args.k <= 0 or args.k > args.train:
        raise SystemExit("--k must be positive and <= --train")


def generate_dataset(
    args: argparse.Namespace,
) -> tuple[
    list[SparseVector],
    list[SparseVector],
    list[list[int]],
]:
    train = make_vectors(
        args.train,
        vocab=args.vocab,
        topics=args.topics,
        nnz=args.nnz,
        seed=args.seed,
    )
    test = make_queries(
        args.test,
        vocab=args.vocab,
        topics=args.topics,
        nnz=args.nnz,
        seed=args.seed ^ 0x9E3779B9,
    )
    neighbors = ground_truth(train, test, args.k)
    return train, test, neighbors


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)

    settings = expected_settings(args)
    if not args.force and matching_outputs(args.output, settings):
        print(f"Reusing SparseMIPS smoke dataset at {args.output}")
        return

    train, test, neighbors = generate_dataset(args)
    train_shape = write_spv1(args.output / "train.spv1", train)
    test_shape = write_spv1(args.output / "test.spv1", test)
    neighbors_shape = write_nbr1(args.output / "neighbors.bin", neighbors)
    write_manifest_atomic(
        args.output / "sparse_mips_dataset.json",
        manifest_for(
            settings,
            train_shape=train_shape,
            test_shape=test_shape,
            neighbors_shape=neighbors_shape,
        ),
    )
    print(f"Wrote SparseMIPS smoke dataset to {args.output}")


if __name__ == "__main__":
    main()
