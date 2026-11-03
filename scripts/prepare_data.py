#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare every dataset a benchmark row in this repo was measured on.

One entry point, two kinds of data, one manifest. Anything a harness reads is
written by this script, so a reader reproduces a row by running the same command
we ran rather than by trusting a size in a README.

Usage:
    uv run scripts/prepare_data.py list
    uv run scripts/prepare_data.py synthetic --points 10m --dim 96
    uv run scripts/prepare_data.py ann-benchmarks glove-25-angular
    uv run scripts/prepare_data.py verify
    uv run scripts/prepare_data.py verify --strict

Layout under the data root (`ANN_DATA_ROOT`, else
`$REBUILDABLE_ROOT/vicinity-data`, else `./data`):

    manifest.json
    datasets/uniform-n10000000-d96.bin      # VICDS01, written by large_scale
    gt/...                                  # ground-truth cache, keyed by query bytes
    ann-benchmarks/<name>/train.bin         # VEC1, written by the downloader

The manifest is the provenance record: path, bytes, sha256, row count, dimension
and the command that produced the file. `verify` recomputes every checksum, so a
silently truncated download is an error rather than a smaller benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = "manifest.json"

# Synthetic domains the tier harness can generate. `all_families` also has
# text/image/sparse/neardup/lowdim generators, but they are fixed at 10k and are
# not yet wired to the file format; listing them here would be a promise the
# script cannot keep.
SYNTHETIC = {
    "uniform": "96-d i.i.d. uniform in [-1,1]^96; the baseline domain",
    "clustered": "16 uniform centres, sigma=0.25; mild LID, 1.4 separation",
}

PUBLIC = {
    "glove-25-angular": "1.18M x 25, angular, 10k queries; the ann-benchmarks classic",
    "glove-100-angular": "1.18M x 100, angular",
    "sift-128-euclidean": "1M x 128, euclidean",
    "gist-960-euclidean": "1M x 960, euclidean",
    "fashion-mnist-784-euclidean": "60k x 784, euclidean",
    "nytimes-16-angular": "290k x 16, angular",
    "lastfm-64-dot": "292k x 64, inner product",
}

# What these benchmarks are and whether they are still alive. `prepare_data.py`
# can only fetch ann-benchmarks today; the rest are named so a reader knows what
# exists and why a given row is not on it.
BENCHMARKS = [
    (
        "ann-benchmarks (erikbern/ann-benchmarks)",
        "alive: repo pushed 2026-07-10, HDF5 files served with HTTP 200 from "
        "ann-benchmarks.com",
        "the classic sets: SIFT, GIST, GloVe, Fashion-MNIST, NYTimes, LastFM",
        "no OOD, no modern embedding models, no int8/binary variants",
    ),
    (
        "VIBE (vector-index-bench/vibe, MIT)",
        "active: 2026 paper, JDCMLR 2026, arXiv:2505.17810",
        "modern embedding sets, 11 in-distribution and 8 OOD, int8 and binary "
        "variants, 22 algorithms, single-core protocol",
        "Linux-only harness, and a GPU track we cannot reproduce on CPU",
    ),
    (
        "big-ann-benchmarks (NeurIPS 2021/2023/2025)",
        "active for competitions",
        "billion-scale, plus filtered, sparse, streaming and OOD tracks",
        "the billion-scale tracks need >1 TB of fast disk; a single-machine "
        "reproduction is not honest",
    ),
    (
        "VectorDBBench (zilliztech)",
        "active",
        "end-to-end database comparison, insertion, filtered and full-text",
        "measures deployments, not indexes; it belongs with our server rows",
    ),
]


def parse_points(text: str) -> int:
    """The same spelling examples/large_scale.rs accepts, resolved to a count."""
    suffix = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    cleaned = text.strip().lower()
    if cleaned and cleaned[-1] in suffix:
        return int(float(cleaned[:-1]) * suffix[cleaned[-1]])
    return int(cleaned)


def data_root() -> Path:
    env = os.environ.get("ANN_DATA_ROOT")
    if env:
        return Path(env)
    rebuildable = os.environ.get("REBUILDABLE_ROOT")
    if rebuildable:
        return Path(rebuildable) / "vicinity-data"
    return REPO / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(root: Path) -> dict:
    path = root / MANIFEST
    if not path.exists():
        return {"schema": 1, "entries": {}}
    return json.loads(path.read_text())


def save_manifest(root: Path, manifest: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def record(
    root: Path, name: str, path: Path, *, rows: int, dim: int, command: str
) -> None:
    manifest = load_manifest(root)
    manifest["entries"][name] = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "rows": rows,
        "dim": dim,
        "command": command,
    }
    save_manifest(root, manifest)
    entry = manifest["entries"][name]
    print(
        f"recorded {name}: {entry['rows']} rows, {entry['bytes']} bytes, {entry['sha256'][:12]}…"
    )


def cmd_list(args) -> int:
    root = data_root()
    manifest = load_manifest(root)
    print(f"data root: {root}")
    print("synthetic (written by examples/large_scale.rs):")
    for name, description in SYNTHETIC.items():
        print(f"  {name:12s} {description}")
    print("public (written by scripts/download_ann_benchmarks.py):")
    for name, description in PUBLIC.items():
        status = "present" if name in manifest["entries"] else "absent"
        print(f"  {name:30s} {status:7s} {description}")
    print("benchmarks and their state:")
    for name, state, gives, limits in BENCHMARKS:
        print(f"  {name}\n    state: {state}\n    gives: {gives}\n    limits: {limits}")
    return 0


def cmd_synthetic(args) -> int:
    root = data_root()
    if args.domain not in SYNTHETIC:
        print(f"unknown domain {args.domain}; see `list`", file=sys.stderr)
        return 2
    if args.domain != "uniform":
        print(
            "only uniform is wired to the file format today; clustered/others are "
            "generated in memory by all_families at 10k",
            file=sys.stderr,
        )
        return 2

    count = parse_points(args.points)
    env = dict(os.environ, ANN_DATA_ROOT=str(root))
    command = [
        "cargo",
        "run",
        "--release",
        "--example",
        "large_scale",
        "--features",
        "large-scale,hnsw,diskann",
        "--",
        "--points",
        str(count),
        "--prepare-only",
    ]
    subprocess.run(command, cwd=REPO, env=env, check=True)
    path = root / "datasets" / f"uniform-n{count}-d{args.dim}.bin"
    if not path.exists():
        print(f"expected {path} after generation; not found", file=sys.stderr)
        return 1
    record(
        root,
        f"synthetic/uniform-n{count}-d{args.dim}",
        path,
        rows=count,
        dim=args.dim,
        command=" ".join(command),
    )
    return 0


def cmd_public(args) -> int:
    root = data_root()
    command = [
        sys.executable,
        str(REPO / "scripts" / "download_ann_benchmarks.py"),
        args.dataset,
    ]
    env = dict(os.environ, ANN_DATA_ROOT=str(root))
    subprocess.run(command, cwd=REPO, env=env, check=True)
    directory = root / "ann-benchmarks" / args.dataset
    train = directory / "train.bin"
    if not train.exists():
        train = REPO / "data" / "ann-benchmarks" / args.dataset / "train.bin"
    if not train.exists():
        print(f"{args.dataset} downloaded but no train.bin found", file=sys.stderr)
        return 1
    # VEC1 header: 4-byte magic, u32 count, u32 dimension.
    with train.open("rb") as handle:
        header = handle.read(12)
    count = int.from_bytes(header[4:8], "little")
    dim = int.from_bytes(header[8:12], "little")
    record(
        root,
        f"ann-benchmarks/{args.dataset}",
        train,
        rows=count,
        dim=dim,
        command=" ".join(command),
    )
    return 0


def cmd_verify(args) -> int:
    root = data_root()
    manifest = load_manifest(root)
    if not manifest["entries"]:
        print("manifest is empty; nothing to verify", file=sys.stderr)
        return 1
    failures = 0
    for name, entry in sorted(manifest["entries"].items()):
        path = root / entry["path"]
        if not path.exists():
            print(f"MISSING {name}: {path}")
            failures += 1
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            print(f"CHANGED {name}: {entry['sha256'][:12]} -> {actual[:12]}")
            failures += 1
        elif args.strict:
            print(f"ok {name}")
    if failures:
        print(f"{failures} entr{'y' if failures == 1 else 'ies'} failed verification")
        return 1
    print(f"verified {len(manifest['entries'])} entries")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser(
        "list", help="what can be prepared, and what each benchmark is"
    )
    listing.set_defaults(func=cmd_list)

    synthetic = sub.add_parser("synthetic", help="generate a synthetic corpus")
    synthetic.add_argument("--domain", default="uniform", choices=sorted(SYNTHETIC))
    synthetic.add_argument("--points", default="10m")
    synthetic.add_argument("--dim", type=int, default=96)
    synthetic.set_defaults(func=cmd_synthetic)

    public = sub.add_parser("ann-benchmarks", help="download and convert a public set")
    public.add_argument("dataset", choices=sorted(PUBLIC))
    public.set_defaults(func=cmd_public)

    verify = sub.add_parser("verify", help="recompute every recorded checksum")
    verify.add_argument("--strict", action="store_true", help="print every entry")
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
