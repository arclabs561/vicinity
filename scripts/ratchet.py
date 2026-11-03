#!/usr/bin/env python3
"""Performance ratchet for the all-family ANN harness.

Compares one run of ``examples/all_families.rs`` against the committed baseline
in ``benchmarks/baseline.jsonl`` and fails when a metric regresses past its
tolerance. Improvements are reported and never fail. ``--update`` rewrites the
baseline, so making the bar higher is an explicit commit whose message can show
the deltas.

The contract is the JSONL emitted by ``all_families --json``::

    {"_meta": true, "schema": 1, "n": 10000, "dim": 96, "k": 10, ...}
    {"family": "hnsw", "dataset": "uniform", "knob": "ef_search",
     "knob_value": 100, "recall_at_10": 0.998, "inverse_ratio": 12.4,
     "p50_us": 82.1, "p95_us": 120.0, "p99_us": 190.0, "qps": 12100.0,
     "build_s": 1.24, "mem_mib": 41.0, "dists_per_query": 812.0}

A row missing from the current run is a failure by default (a family silently
dropping out is a regression the metric table cannot show); ``--allow-subset``
turns that into a warning, and ``--family`` implies it.

Examples::

    # Check the working tree against the committed baseline.
    scripts/ratchet.py

    # Compare two pre-recorded runs (no rebuild).
    scripts/ratchet.py --current /tmp/rows.jsonl

    # Raise the bar after a real improvement.
    scripts/ratchet.py --update --save-current /tmp/rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "benchmarks" / "baseline.jsonl"

# Families the harness can drive. Keep in sync with the `[[example]]`
# `required-features` list and `families()` in `examples/all_families.rs`.
FAMILIES = [
    "hnsw",
    "nsw",
    "nsg",
    "finger",
    "fresh_graph",
    "filtered_graph",
    "vamana",
    "diskann",
    "ivf_pq",
    "ivf_avq",
    "ivf_rabitq",
    "emg",
    "curator",
    "lsh",
    "pipnn",
    "rp_quant",
    "sq4",
    "binary_index",
    "sng",
    "range_filtered",
    "kdtree",
    "balltree",
    "kmeans_tree",
    "rptree",
]


@dataclass(frozen=True)
class Metric:
    """One gated measurement.

    ``tolerance`` is the ratio a value may grow by before it is a regression
    (``dists_per_query``, ``p50_us``, ...). ``absolute_tolerance`` is used for
    recall, where a ratio of a number near 1 is not meaningful.
    ``min_absolute`` is a noise floor: below it the measurement cannot tell a
    change from run-to-run jitter (4 µs vs 8 µs on a 200-query loop), so the
    row is not gated on this metric.
    """

    field: str
    label: str
    higher_is_better: bool
    tolerance: float
    absolute_tolerance: float | None = None
    min_absolute: float = 0.0

    def delta(self, baseline: float, current: float) -> float:
        return current - baseline

    def regressed(self, baseline: float, current: float) -> bool:
        if self.absolute_tolerance is not None:
            if self.higher_is_better:
                return baseline - current > self.absolute_tolerance
            return current - baseline > self.absolute_tolerance
        if baseline <= 0:
            # No ratio is possible against zero; `compare` reports it as
            # unmeasured instead of failing on any positive value.
            return False
        movement = (
            current - baseline if not self.higher_is_better else baseline - current
        )
        if movement < self.min_absolute:
            return False
        if self.higher_is_better:
            return current < baseline * (1.0 - self.tolerance)
        return current > baseline * (1.0 + self.tolerance)

    def improved(self, baseline: float, current: float) -> bool:
        """True when the value moved in the good direction by a visible amount.

        This uses a fixed 1% (or 0.001 absolute) reporting margin, not the
        regression tolerance: a 10% latency win is worth printing even though
        the gate would tolerate a 15% loss.
        """
        if self.absolute_tolerance is not None:
            margin = 0.001
            if self.higher_is_better:
                return current - baseline > margin
            return baseline - current > margin
        if baseline <= 0:
            return False
        if self.higher_is_better:
            improved = current > baseline * 1.01
        else:
            improved = current < baseline * 0.99
        return improved and abs(current - baseline) >= self.min_absolute

    def limit(self, baseline: float) -> str:
        if self.absolute_tolerance is not None:
            sign = "-" if self.higher_is_better else "+"
            return f"{sign}{self.absolute_tolerance:g} abs"
        text = f"{self.tolerance * 100:.0f}%"
        if self.min_absolute > 0:
            text += f" (floor {self.min_absolute:g})"
        return text


# The gate, in the order it is reported.
METRICS = [
    Metric("recall_at_10", "recall@10", True, 0.0, absolute_tolerance=0.005),
    Metric("p50_us", "p50", False, 0.15, min_absolute=10.0),
    Metric("p99_us", "p99", False, 0.25, min_absolute=50.0),
    Metric("dists_per_query", "dists/q", False, 0.20, min_absolute=5.0),
    Metric("mem_mib", "mem MiB", False, 0.15, min_absolute=2.0),
    Metric("build_s", "build s", False, 0.25, min_absolute=0.01),
]

KEY_FIELDS = ("family", "dataset", "knob", "knob_value")


def key_of(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return tuple(row[field] for field in KEY_FIELDS)  # type: ignore[return-value]


def describe(key: tuple[str, str, str, int]) -> str:
    family, dataset, knob, value = key
    return f"{family} / {dataset} / {knob}={value}"


def read_jsonl(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Return (meta, rows, coverage-failure notes)."""
    if not path.exists():
        raise SystemExit(f"ratchet: no such file: {path}")
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"ratchet: {path}:{line_number}: invalid JSON: {error}"
                ) from error
            if record.get("_meta"):
                meta = record
            elif record.get("_failure"):
                failures.append(
                    f"{record.get('family', '?')}: {record.get('error', '?')}"
                )
            else:
                # The harness has emitted the recall field under both names:
                # `recall_at_10` in the committed baseline, `recall` since the
                # row schema was widened. Normalise on load so one gate reads
                # both, rather than silently comparing against None and calling
                # every row a regression.
                if "recall_at_10" not in record and "recall" in record:
                    record["recall_at_10"] = record["recall"]
                rows.append(record)
    return meta, rows, failures


def index_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        missing = [field for field in KEY_FIELDS if field not in row]
        if missing:
            raise SystemExit(f"ratchet: row is missing key fields {missing}: {row}")
        key = key_of(row)
        if key in indexed:
            print(
                f"ratchet: duplicate row for {describe(key)}; the last one wins",
                file=sys.stderr,
            )
        indexed[key] = row
    return indexed


def reduce_runs(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Per-row median across repeated runs.

    Wall clock on a shared machine is noisy at the tail: one scheduling hiccup
    can move a single run's p99 by an order of magnitude. A median over three
    runs keeps a regression (which moves every run) and drops a one-run spike.
    """
    if len(runs) == 1:
        return runs[0]
    indexed = [index_rows(run) for run in runs]
    keys = sorted({key for run in indexed for key in run})
    merged: list[dict[str, Any]] = []
    for key in keys:
        present = [run[key] for run in indexed if key in run]
        if len(present) < len(runs):
            print(
                f"ratchet: {describe(key)} appears in {len(present)}/{len(runs)} runs",
                file=sys.stderr,
            )
        row = dict(present[0])
        for metric in METRICS:
            values = [
                value
                for value in (numeric(r.get(metric.field)) for r in present)
                if value is not None
            ]
            row[metric.field] = statistics.median(values) if values else None
        merged.append(row)
    return merged


def harness_command(family: str | None) -> list[str]:
    features = ",".join([*FAMILIES, "distance-stats"])
    command = [
        "cargo",
        "run",
        "--release",
        "--example",
        "all_families",
        "--features",
        features,
        "--",
        "--json",
    ]
    if family:
        command.append(family)
    return command


def run_harness(family: str | None, save_current: Path | None) -> Path:
    """Run the harness, streaming its stderr, and return the JSONL path."""
    command = harness_command(family)
    print(f"ratchet: running `{' '.join(command)}`", file=sys.stderr)
    if save_current:
        save_current.parent.mkdir(parents=True, exist_ok=True)
        sink = save_current
        handle = sink.open("w", encoding="utf-8")
    else:
        descriptor, name = tempfile.mkstemp(prefix="ratchet-rows-", suffix=".jsonl")
        sink = Path(name)
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
    with handle:
        completed = subprocess.run(command, stdout=handle, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"ratchet: harness exited {completed.returncode}")
    return sink


def numeric(value: Any) -> float | None:
    """A baseline/current cell as a float, or None when absent/not measured."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def format_value(field: str, value: float | None) -> str:
    if value is None:
        return "—"
    if field == "recall_at_10":
        return f"{value:.4f}"
    if field in {"p50_us", "p99_us"}:
        return f"{value:.1f}"
    if field == "dists_per_query":
        return f"{value:.0f}"
    if field == "mem_mib":
        return f"{value:.1f}"
    if field == "build_s":
        return f"{value:.3f}"
    return f"{value:.3f}"


def format_delta(metric: Metric, baseline: float, current: float) -> str:
    difference = metric.delta(baseline, current)
    # A relative delta on a number near 1 understates a recall change.
    if metric.absolute_tolerance is not None:
        return f"{difference:+.4f}"
    if baseline == 0:
        return f"{difference:+.4g}"
    return f"{difference / baseline * 100:+.1f}%"


def load_ratio(meta: dict[str, Any]) -> float | None:
    """One-minute load per core, or None when the run did not record it."""
    load = meta.get("load1")
    cores = meta.get("cores")
    if isinstance(load, (int, float)) and isinstance(cores, int) and cores > 0:
        return float(load) / cores
    return None


def compare(
    baseline: dict[tuple[str, str, str, int], dict[str, Any]],
    current: dict[tuple[str, str, str, int], dict[str, Any]],
    *,
    allow_subset: bool,
    advisory_timing: bool,
    advisory_reason: str | None = None,
) -> int:
    failures: list[tuple[str, str, str, str, str, str]] = []
    improvements: list[tuple[str, str, str, str, str, str]] = []
    missing: list[str] = []
    unavailable: list[str] = []
    checked = 0

    for key in sorted(baseline):
        base_row = baseline[key]
        current_row = current.get(key)
        if current_row is None:
            missing.append(describe(key))
            continue
        for metric in METRICS:
            base_value = numeric(base_row.get(metric.field))
            current_value = numeric(current_row.get(metric.field))
            if base_value is None or current_value is None:
                if base_value is not None or current_value is not None:
                    unavailable.append(f"{describe(key)}: {metric.label}")
                continue
            if metric.absolute_tolerance is None and base_value <= 0:
                unavailable.append(
                    f"{describe(key)}: {metric.label} (baseline 0; no ratio)"
                )
                continue
            if metric.regressed(base_value, current_value):
                advisory = (
                    advisory_timing
                    and metric.field != "dists_per_query"
                    and metric.field != "recall_at_10"
                )
                status = "advisory" if advisory else "fail"
                failures.append(
                    (
                        status,
                        describe(key),
                        metric.label,
                        format_value(metric.field, base_value),
                        format_value(metric.field, current_value),
                        format_delta(metric, base_value, current_value)
                        + f" (limit {metric.limit(base_value)})",
                    )
                )
            elif metric.improved(base_value, current_value):
                improvements.append(
                    (
                        "improved",
                        describe(key),
                        metric.label,
                        format_value(metric.field, base_value),
                        format_value(metric.field, current_value),
                        format_delta(metric, base_value, current_value),
                    )
                )
            checked += 1

    added = sorted(set(current) - set(baseline))

    print(
        f"ratchet: compared {len(baseline)} baseline rows against "
        f"{len(current)} current rows ({checked} metric comparisons)"
    )
    if added:
        print(f"ratchet: {len(added)} row(s) are new since the baseline (not gated):")
        for key in added:
            print(f"  + {describe(key)}")

    if advisory_reason:
        print("\n## Timing gates not applied")
        print(f"- {advisory_reason}")
        print(
            "- recall@10 and dists/query are hardware-independent and still gate; "
            "p50/p99/mem/build are printed above only when they moved."
        )

    if unavailable:
        print(f"\n## Not measured ({len(unavailable)})")
        for note in unavailable:
            print(f"- {note}")

    if improvements:
        print(f"\n## Improvements ({len(improvements)}) — reported, never failed")
        print("| row | metric | baseline | current | delta |")
        print("| --- | --- | ---: | ---: | ---: |")
        for _, row, label, base_text, current_text, delta in improvements:
            print(f"| {row} | {label} | {base_text} | {current_text} | {delta} |")

    hard_failures = [entry for entry in failures if entry[0] == "fail"]
    advisories = [entry for entry in failures if entry[0] == "advisory"]

    if advisories:
        print(f"\n## Timing advisories ({len(advisories)}) — not failing this run")
        print("| row | metric | baseline | current | delta |")
        print("| --- | --- | ---: | ---: | ---: |")
        for _, row, label, base_text, current_text, delta in advisories:
            print(f"| {row} | {label} | {base_text} | {current_text} | {delta} |")

    if missing:
        stream = sys.stderr if allow_subset else None
        header = f"\n## Missing rows ({len(missing)})"
        print(header, file=stream or sys.stdout)
        for row in missing:
            print(f"- {row}", file=stream or sys.stdout)
        if not allow_subset:
            hard_failures.append(
                (
                    "fail",
                    "coverage",
                    "missing row",
                    "present",
                    "absent",
                    "run the full harness or pass --allow-subset",
                )
            )

    if hard_failures:
        print(f"\n## Ratchet failures ({len(hard_failures)})")
        print("| row | metric | baseline | current | delta |")
        print("| --- | --- | ---: | ---: | ---: |")
        for _, row, label, base_text, current_text, delta in hard_failures:
            print(f"| {row} | {label} | {base_text} | {current_text} | {delta} |")
        print(
            "\nratchet: FAILED — a metric regressed past its tolerance "
            "or coverage dropped."
        )
        return 1

    print("\nratchet: PASS — no metric regressed past its tolerance.")
    return 0


def update_baseline(
    baseline_path: Path,
    current_rows: list[dict[str, Any]],
    meta: dict[str, Any],
    previous: dict[tuple[str, str, str, int], dict[str, Any]],
    *,
    allow_subset: bool,
) -> int:
    """Rewrite the baseline and print the deltas it encodes."""
    if not current_rows:
        raise SystemExit("ratchet: refusing to write an empty baseline")

    current = index_rows(current_rows)
    dropped = sorted(set(previous) - set(current))
    if dropped and not allow_subset:
        raise SystemExit(
            f"ratchet: refusing to drop {len(dropped)} baseline row(s) without "
            "--allow-subset; run the full harness, or pass --allow-subset to "
            "shrink the baseline deliberately"
        )
    changes: list[tuple[str, str, str, str, str]] = []
    for key in sorted(current):
        for metric in METRICS:
            new_value = numeric(current[key].get(metric.field))
            old_value = numeric(previous.get(key, {}).get(metric.field))
            if new_value is None or old_value is None:
                continue
            if metric.regressed(old_value, new_value) or metric.improved(
                old_value, new_value
            ):
                changes.append(
                    (
                        describe(key),
                        metric.label,
                        format_value(metric.field, old_value),
                        format_value(metric.field, new_value),
                        format_delta(metric, old_value, new_value),
                    )
                )

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = baseline_path.with_suffix(baseline_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if meta:
            handle.write(
                json.dumps(
                    {"_meta": True, **{k: v for k, v in meta.items() if k != "_meta"}},
                    sort_keys=True,
                )
            )
            handle.write("\n")
        for key in sorted(current):
            handle.write(json.dumps(current[key], sort_keys=True))
            handle.write("\n")
    temporary.replace(baseline_path)

    print(f"ratchet: baseline written to {baseline_path} ({len(current)} rows)")
    if changes:
        print(
            f"\n## Baseline deltas ({len(changes)}) — quote these in the commit message"
        )
        print("| row | metric | old | new | delta |")
        print("| --- | --- | ---: | ---: | ---: |")
        for row, label, old_text, new_text, delta in changes:
            print(f"| {row} | {label} | {old_text} | {new_text} | {delta} |")
    else:
        print(
            "ratchet: no metric moved past its tolerance; "
            "this is a re-recording, not a bar change"
        )
    print(
        "\nratchet: commit `benchmarks/baseline.jsonl` with the deltas above. "
        "A baseline change without a delta in the message is not reviewable."
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail when an all-family ANN run regresses past the committed baseline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="baseline JSONL (default: benchmarks/baseline.jsonl)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        action="append",
        help=(
            "compare this JSONL instead of running the harness; repeat it to "
            "take the per-row median of several runs"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "run the harness N times and compare the per-row median "
            "(default: 1; 3 is the reliable setting for timing gates)"
        ),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from the current run",
    )
    parser.add_argument("--family", help="run one family (implies --allow-subset)")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="missing rows are warnings, not failures",
    )
    parser.add_argument(
        "--advisory-timing",
        action="store_true",
        help=(
            "report timing/memory regressions without failing "
            "(use only across heterogeneous hosts)"
        ),
    )
    parser.add_argument(
        "--max-load-ratio",
        type=float,
        default=1.0,
        help=(
            "treat timings as advisory when a run's 1-minute load per core "
            "exceeds this (default: 1.0; 0 disables the check)"
        ),
    )
    parser.add_argument(
        "--save-current",
        type=Path,
        help="keep the JSONL that was compared (artifact, or input to --update)",
    )
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=0.005,
        help="absolute recall@10 drop allowed (default: 0.005)",
    )
    parser.add_argument(
        "--p50-tolerance",
        type=float,
        default=0.15,
        help="relative p50 growth allowed (default: 0.15)",
    )
    parser.add_argument(
        "--p99-tolerance",
        type=float,
        default=0.25,
        help="relative p99 growth allowed (default: 0.25)",
    )
    parser.add_argument(
        "--dists-tolerance",
        type=float,
        default=0.20,
        help="relative dists/query growth allowed (default: 0.20)",
    )
    parser.add_argument(
        "--mem-tolerance",
        type=float,
        default=0.15,
        help="relative mem growth allowed (default: 0.15)",
    )
    parser.add_argument(
        "--build-tolerance",
        type=float,
        default=0.25,
        help="relative build-time growth allowed (default: 0.25)",
    )
    return parser.parse_args(argv)


def apply_tolerances(args: argparse.Namespace) -> None:
    overrides = {
        "recall_at_10": args.recall_tolerance,
        "p50_us": args.p50_tolerance,
        "p99_us": args.p99_tolerance,
        "dists_per_query": args.dists_tolerance,
        "mem_mib": args.mem_tolerance,
        "build_s": args.build_tolerance,
    }
    for index, metric in enumerate(METRICS):
        value = overrides[metric.field]
        if metric.absolute_tolerance is not None:
            METRICS[index] = Metric(
                metric.field,
                metric.label,
                metric.higher_is_better,
                metric.tolerance,
                value,
                metric.min_absolute,
            )
        else:
            METRICS[index] = Metric(
                metric.field,
                metric.label,
                metric.higher_is_better,
                value,
                None,
                metric.min_absolute,
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_tolerances(args)

    allow_subset = args.allow_subset or bool(args.family)

    coverage_notes: list[str] = []
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    if args.current:
        runs = []
        for path in args.current:
            run_meta, run_rows, run_notes = read_jsonl(path)
            meta = meta or run_meta
            runs.append(run_rows)
            coverage_notes.extend(run_notes)
        if args.save_current and len(args.current) == 1:
            source = args.current[0]
            if args.save_current != source:
                args.save_current.parent.mkdir(parents=True, exist_ok=True)
                args.save_current.write_text(
                    source.read_text(encoding="utf-8"), encoding="utf-8"
                )
        rows = reduce_runs(runs)
    else:
        produced_runs: list[list[dict[str, Any]]] = []
        for repeat in range(args.runs):
            if args.runs > 1:
                print(
                    f"ratchet: run {repeat + 1}/{args.runs}",
                    file=sys.stderr,
                )
            produced = run_harness(
                args.family,
                args.save_current if args.runs == 1 else None,
            )
            run_meta, run_rows, run_notes = read_jsonl(produced)
            meta = meta or run_meta
            produced_runs.append(run_rows)
            coverage_notes.extend(run_notes)
            if args.save_current is None:
                produced.unlink(missing_ok=True)
        rows = reduce_runs(produced_runs)

    for note in coverage_notes:
        print(f"ratchet: harness could not run {note}", file=sys.stderr)

    if not rows:
        raise SystemExit("ratchet: the current run produced no rows")

    baseline_meta: dict[str, Any] = {}
    baseline_rows: list[dict[str, Any]] = []
    if args.baseline.exists():
        baseline_meta, baseline_rows, _ = read_jsonl(args.baseline)
    elif not args.update:
        raise SystemExit(
            f"ratchet: no baseline at {args.baseline}; "
            "run `scripts/ratchet.py --update` to create one"
        )
    previous = index_rows(baseline_rows)

    if args.update:
        if args.family:
            raise SystemExit(
                "ratchet: --update needs a full run; drop --family (or use "
                "--allow-subset to shrink the baseline deliberately)"
            )
        return update_baseline(
            args.baseline, rows, meta, previous, allow_subset=args.allow_subset
        )

    if baseline_meta.get("schema") != meta.get("schema"):
        print(
            f"ratchet: baseline schema {baseline_meta.get('schema')} "
            f"!= current {meta.get('schema')}",
            file=sys.stderr,
        )

    baseline_host = baseline_meta.get("host")
    current_host = meta.get("host")
    if baseline_host and current_host and baseline_host != current_host:
        print(
            f"ratchet: host mismatch — baseline {baseline_host}, "
            f"current {current_host}.",
            file=sys.stderr,
        )
        print(
            "ratchet: recall and dists/query are hardware-independent and "
            "still gate; timings and memory move with the machine. Pass "
            "--advisory-timing to demote them to advisories.",
            file=sys.stderr,
        )
    if baseline_meta.get("distance_stats") and not meta.get("distance_stats"):
        print(
            "ratchet: baseline has dists/query but this run was built without "
            "`distance-stats`; that metric is not gated.",
            file=sys.stderr,
        )

    contended: list[str] = []
    if args.max_load_ratio > 0:
        for label, run_meta in (("baseline", baseline_meta), ("current", meta)):
            ratio = load_ratio(run_meta)
            if ratio is not None and ratio > args.max_load_ratio:
                contended.append(
                    f"{label} recorded load {run_meta.get('load1')} on "
                    f"{run_meta.get('cores')} cores ({ratio:.1f}x)"
                )
    advisory_reason = None
    if contended:
        advisory_reason = "; ".join(contended) + (
            f", above --max-load-ratio {args.max_load_ratio:g}. Wall clock here "
            "measures the scheduler, so p50/p99/mem/build are advisory."
        )
        print(f"ratchet: {advisory_reason}", file=sys.stderr)

    return compare(
        previous,
        index_rows(rows),
        allow_subset=allow_subset,
        advisory_timing=args.advisory_timing or bool(contended),
        advisory_reason=advisory_reason,
    )


if __name__ == "__main__":
    sys.exit(main())
