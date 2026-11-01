#!/usr/bin/env python3
"""Per-method benefit profile from benchmark rows.

Every method wins somewhere: a brute-force tree is exact but slow, an IVF scan
is cheap to build but needs training, a graph has the best recall-per-query but
no exactness. This script makes those advantages legible instead of leaving a
single recall/QPS number to stand for "best".

Input: JSONL, one object per measurement, fields:
  family, dataset, knob, knob_value, recall, p50_us, p95_us, p99_us, qps,
  build_s, mem_mib (nullable), dists_per_query (nullable), exact (bool),
  supports_update (bool), supports_filter (bool)

Output: per family, the best row per dataset; per axis, the leader; the Pareto
front on (recall, p50); a one-line "wins when" statement per family.
"""

import json
import sys
from collections import defaultdict

AXES = [
    ("recall", "highest recall at any budget", True),
    ("qps", "fastest at its best recall", True),
    ("build_s", "cheapest to build", False),
    ("mem_mib", "smallest resident growth", False),
    ("dists_per_query", "fewest distance computations", False),
    ("tail_ratio", "best tail (p99/p50)", False),
]

def tail_ratio(row):
    p50 = row.get("p50_us") or 0.0
    p99 = row.get("p99_us")
    if not p50 or p99 is None:
        return None
    return p99 / p50

def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def best_per_family(rows):
    best = {}
    for row in rows:
        key = (row["family"], row["dataset"])
        current = best.get(key)
        if current is None or row["recall"] > current["recall"] or (
            row["recall"] == current["recall"] and row["p50_us"] < current["p50_us"]
        ):
            best[key] = row
    return best

def pareto(rows):
    front = []
    for row in sorted(rows, key=lambda r: (-r["recall"], r["p50_us"])):
        if not any(
            other["recall"] >= row["recall"] and other["p50_us"] <= row["p50_us"]
            and (other["recall"] > row["recall"] or other["p50_us"] < row["p50_us"])
            for other in rows
        ):
            front.append(row)
    return front

def leader(rows, key, higher_is_better, transform=None):
    scored = []
    for row in rows:
        value = transform(row) if transform else row.get(key)
        if value is None:
            continue
        scored.append((value, row))
    if not scored:
        return None, None
    pick = max(scored, key=lambda p: p[0]) if higher_is_better else min(scored, key=lambda p: p[0])
    return pick[1], pick[0]

def wins_when(row):
    notes = []
    if row.get("exact"):
        notes.append("exact answers")
    if row.get("supports_update"):
        notes.append("updates in place")
    if row.get("supports_filter"):
        notes.append("filtered search")
    if row.get("mem_mib") is not None:
        notes.append(f"{row['mem_mib']:.0f} MiB resident")
    if row.get("dists_per_query") is not None:
        notes.append(f"{row['dists_per_query']:.0f} distances/query")
    if row.get("build_s") is not None:
        notes.append(f"{row['build_s']:.1f} s to build")
    return ", ".join(notes) if notes else "measured, no distinguishing axis"

def main(path):
    rows = load(path)
    best = best_per_family(rows)
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for (family, ds), row in best.items() if ds == dataset]
        print(f"## {dataset}")
        print()
        print("| family | at | recall | p50 us | qps | build s | mem MiB | dists/q | wins when |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in sorted(subset, key=lambda r: -r["recall"]):
            mem = row.get("mem_mib")
            dists = row.get("dists_per_query")
            print(
                f"| {row['family']} | {row.get('knob','?')}={row.get('knob_value','?')} | "
                f"{row['recall']:.3f} | {row['p50_us']:.1f} | {row['qps']:.0f} | "
                f"{row.get('build_s', 0):.1f} | "
                f"{'-' if mem is None else f'{mem:.1f}'} | "
                f"{'-' if dists is None else f'{dists:.0f}'} | {wins_when(row)} |"
            )
        print()
        print("### axis leaders")
        print()
        for key, label, higher in AXES:
            transform = tail_ratio if key == "tail_ratio" else None
            row, value = leader(subset, key, higher, transform)
            if row is not None:
                shown = f"{value:.3g}" if isinstance(value, float) else str(value)
                print(f"- **{label}**: `{row['family']}` ({shown})")
        front = pareto(subset)
        print()
        print(
            "### Pareto front on (recall, p50): "
            + ", ".join(f"`{r['family']}` {r['recall']:.3f}@{r['p50_us']:.0f}us" for r in front)
        )
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: benefits.py <rows.jsonl>")
    main(sys.argv[1])
