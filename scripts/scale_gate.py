#!/usr/bin/env python3
"""Recall gate for a `large_scale` run against committed reference rows.

Timing is hardware-dependent, so this compares only what is not: recall at a
fixed knob value, and the set of rows that exist. HNSW builds are seeded, so a
machine with the same seed produces a graph of the same quality; a recall drop
means the code changed, not the runner.

The contract is the JSONL emitted by ``examples/large_scale --json``::

    {"_meta": true, "harness": "large_scale", "n": 1000000, "dim": 96, ...}
    {"family": "hnsw", "dataset": "uniform@1m", "knob": "ef_search",
     "knob_value": 200, "recall_at_10": 0.2936, "p50_us": 922.1, ...}

Examples::

    # Gate a fresh run against the committed 1M row.
    scripts/scale_gate.py --current /tmp/rows.jsonl \
        --reference benchmarks/large-scale-1m-uniform.jsonl

    # Allow a 0.02 absolute drop while a known-bad build is investigated.
    scripts/scale_gate.py --current /tmp/rows.jsonl --tolerance 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_REFERENCE = Path("benchmarks/large-scale-1m-uniform.jsonl")
DEFAULT_TOLERANCE = 0.005


def read_rows(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_meta"):
                meta = row
            else:
                rows.append(row)
    return meta, rows


def key_of(row: dict) -> tuple[str, str, str, int | float]:
    knob_value = row.get("knob_value")
    try:
        knob_value = int(knob_value)
    except (TypeError, ValueError):
        pass
    return (
        row.get("family", ""),
        row.get("dataset", ""),
        row.get("knob", ""),
        knob_value,
    )


def recall_of(row: dict) -> float | None:
    value = row.get("recall_at_10", row.get("recall"))
    return None if value is None else float(value)


def describe(key: tuple[str, str, str, int | float]) -> str:
    family, dataset, knob, value = key
    return f"{family} {dataset} {knob}={value}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument(
        "--require-n",
        type=int,
        default=None,
        help="fail unless the current meta declares this row count",
    )
    args = parser.parse_args(argv)

    if not args.current.exists():
        print(f"scale-gate: missing current rows: {args.current}", file=sys.stderr)
        return 2
    if not args.reference.exists():
        print(f"scale-gate: missing reference rows: {args.reference}", file=sys.stderr)
        return 2

    current_meta, current_rows = read_rows(args.current)
    reference_meta, reference_rows = read_rows(args.reference)

    if args.require_n is not None and current_meta.get("n") != args.require_n:
        print(
            f"scale-gate: current run is n={current_meta.get('n')}, expected n={args.require_n}",
            file=sys.stderr,
        )
        return 2
    if current_meta.get("n") != reference_meta.get("n"):
        print(
            "scale-gate: current n="
            f"{current_meta.get('n')} does not match reference n={reference_meta.get('n')}; "
            "compare against the reference for the same tier",
            file=sys.stderr,
        )
        return 2

    by_key = {key_of(row): row for row in current_rows}
    failures: list[str] = []
    lines = [
        f"| row | reference | current | delta | verdict |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    for reference in reference_rows:
        key = key_of(reference)
        want = recall_of(reference)
        current = by_key.get(key)
        if current is None:
            failures.append(f"missing row: {describe(key)}")
            lines.append(f"| {describe(key)} | {want:.4f} | - | - | missing |")
            continue
        got = recall_of(current)
        if want is None or got is None:
            lines.append(f"| {describe(key)} | - | - | - | unmeasured |")
            continue
        delta = got - want
        if delta < -args.tolerance:
            failures.append(
                f"recall regressed: {describe(key)} {want:.4f} -> {got:.4f}"
            )
            verdict = "regressed"
        else:
            verdict = "ok"
        lines.append(
            f"| {describe(key)} | {want:.4f} | {got:.4f} | {delta:+.4f} | {verdict} |"
        )

    print("\n".join(lines))
    print(
        f"\nreference {args.reference} (host {reference_meta.get('host', '?')}, "
        f"n={reference_meta.get('n')}); tolerance {args.tolerance:.4f} absolute",
    )
    if failures:
        print("\nscale-gate: FAIL", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("scale-gate: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
