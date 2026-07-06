#!/usr/bin/env python3
"""Generate a tiny deterministic VEC1/NBR1 dataset for benchmark smoke tests."""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k > args.train:
        raise SystemExit("--k must be <= --train")

    args.output.mkdir(parents=True, exist_ok=True)
    train = make_train(args.train, args.dim)
    test = make_test(train, args.test)
    neighbors = ground_truth(train, test, args.k)

    write_vec1(args.output / "train.bin", train)
    write_vec1(args.output / "test.bin", test)
    write_nbr1(args.output / "neighbors.bin", neighbors)
    manifest = {
        "version": 1,
        "train": args.train,
        "test": args.test,
        "dim": args.dim,
        "k": args.k,
    }
    (args.output / "ann_smoke_dataset.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote ANN smoke dataset to {args.output}")


if __name__ == "__main__":
    main()
