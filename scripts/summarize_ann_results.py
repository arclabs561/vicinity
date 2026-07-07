#!/usr/bin/env python3
"""Summarize ann_benchmark JSONL coverage by dataset, algorithm, and storage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Summary:
    rows: int = 0
    best_recall: float = 0.0
    best_qps: float = 0.0

    def add(self, row: dict[str, Any]) -> None:
        self.rows += 1
        self.best_recall = max(self.best_recall, float(row.get("recall_at_10", 0.0)))
        self.best_qps = max(self.best_qps, float(row.get("qps", 0.0)))


@dataclass(frozen=True)
class CoverageRow:
    dataset: str
    algorithm: str
    storage_mode: str
    status: str
    rows: int
    best_recall: float | None
    best_qps: float | None


def row_dataset(row: dict[str, Any], current_dataset: str | None, path: Path) -> str:
    dataset = row.get("dataset") or current_dataset
    if isinstance(dataset, str) and dataset:
        return Path(dataset).name
    return path.stem


def load_summaries(paths: list[Path]) -> dict[tuple[str, str, str], Summary]:
    summaries: dict[tuple[str, str, str], Summary] = defaultdict(Summary)
    for path in paths:
        current_dataset: str | None = None
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "_meta" in row:
                    meta = row["_meta"]
                    if isinstance(meta, dict):
                        current_dataset = str(meta.get("dataset") or "") or None
                    continue
                algorithm = row.get("algorithm")
                if not isinstance(algorithm, str):
                    continue
                storage_mode = row.get("storage_mode")
                if not isinstance(storage_mode, str):
                    storage_mode = "in_memory"
                dataset = row_dataset(row, current_dataset, path)
                summaries[(dataset, algorithm, storage_mode)].add(row)
    return dict(summaries)


def coverage_rows(
    summaries: dict[tuple[str, str, str], Summary],
    expected: list[tuple[str, str]] | None = None,
    datasets: list[str] | None = None,
) -> list[CoverageRow]:
    expected = expected or []
    dataset_names = sorted(datasets or {dataset for dataset, _, _ in summaries})
    keys = set(summaries)
    if expected:
        for dataset in dataset_names:
            for algorithm, storage_mode in expected:
                keys.add((dataset, algorithm, storage_mode))

    rows = []
    for dataset, algorithm, storage_mode in sorted(keys):
        summary = summaries.get((dataset, algorithm, storage_mode))
        rows.append(
            CoverageRow(
                dataset=dataset,
                algorithm=algorithm,
                storage_mode=storage_mode,
                status="measured" if summary else "missing",
                rows=summary.rows if summary else 0,
                best_recall=summary.best_recall if summary else None,
                best_qps=summary.best_qps if summary else None,
            )
        )
    return rows


def markdown_table(rows: list[CoverageRow]) -> str:
    lines = [
        "| Dataset | Algorithm | Storage | Status | Rows | Best Recall@10 | Best QPS |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        recall = "-" if row.best_recall is None else f"{row.best_recall:.4f}"
        qps = "-" if row.best_qps is None else f"{row.best_qps:.1f}"
        lines.append(
            f"| {row.dataset} | {row.algorithm} | {row.storage_mode} | "
            f"{row.status} | {row.rows} | {recall} | {qps} |"
        )
    return "\n".join(lines)


def parse_expected(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected row must be algorithm:storage_mode")
    algorithm, storage_mode = value.split(":", 1)
    if not algorithm or not storage_mode:
        raise argparse.ArgumentTypeError("expected row must be algorithm:storage_mode")
    return algorithm, storage_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=list(Path("data/ann-benchmarks/results").glob("*.jsonl")),
        help="ann_benchmark JSONL files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable summary rows instead of markdown",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        type=parse_expected,
        metavar="ALGORITHM:STORAGE",
        help="Mark an expected algorithm/storage row as missing when absent",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to use for expected rows when no measured row exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.paths)
    rows = coverage_rows(summaries, args.expect, args.dataset)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "dataset": row.dataset,
                        "algorithm": row.algorithm,
                        "storage_mode": row.storage_mode,
                        "status": row.status,
                        "rows": row.rows,
                        "best_recall_at_10": row.best_recall,
                        "best_qps": row.best_qps,
                    }
                    for row in rows
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
