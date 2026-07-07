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


def markdown_table(summaries: dict[tuple[str, str, str], Summary]) -> str:
    lines = [
        "| Dataset | Algorithm | Storage | Rows | Best Recall@10 | Best QPS |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for (dataset, algorithm, storage_mode), summary in sorted(summaries.items()):
        lines.append(
            f"| {dataset} | {algorithm} | {storage_mode} | {summary.rows} | "
            f"{summary.best_recall:.4f} | {summary.best_qps:.1f} |"
        )
    return "\n".join(lines)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.paths)
    if args.json:
        rows = [
            {
                "dataset": dataset,
                "algorithm": algorithm,
                "storage_mode": storage_mode,
                "rows": summary.rows,
                "best_recall_at_10": summary.best_recall,
                "best_qps": summary.best_qps,
            }
            for (dataset, algorithm, storage_mode), summary in sorted(summaries.items())
        ]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return
    print(markdown_table(summaries))


if __name__ == "__main__":
    main()
