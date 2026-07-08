#!/usr/bin/env python3
"""Summarize filtered-search selectivity JSONL outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def recall_value(row: dict[str, Any]) -> tuple[str, float | None]:
    for key, value in row.items():
        if key.startswith("recall_at_") and isinstance(value, (int, float)):
            return key, float(value)
    return "recall", None


def number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def fmt_qps(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.1f}"


@dataclass(frozen=True)
class SelectivityRow:
    workload: str
    algorithm: str
    storage_mode: str
    cache_state: str | None
    selectivity: float | None
    target_count: int | None
    recall_key: str
    recall: float | None
    qps: float | None
    index_bytes: int | None
    index_bytes_required: bool
    p95_us: float | None
    p99_us: float | None
    mean_returned: float | None
    two_hop_invocations: float | None
    two_hop_nodes_examined: float | None


def scoped_workload(meta: dict[str, Any], path: Path) -> str:
    workload = meta.get("workload")
    if not isinstance(workload, str) or not workload:
        workload = path.stem
    scope = []
    for key in (
        "n",
        "dim",
        "queries",
        "k",
        "neighbors",
        "ef_search",
        "acorn_max_two_hop_neighbors",
        "fallback_selectivity_threshold",
    ):
        value = meta.get(key)
        if isinstance(value, int):
            scope.append(f"{key}={int(value)}")
        elif isinstance(value, float):
            scope.append(f"{key}={value:g}")
    if scope:
        return f"{workload}[{','.join(scope)}]"
    return workload


def parse_row(
    row: dict[str, Any], workload: str, *, index_bytes_required: bool = False
) -> SelectivityRow | None:
    algorithm = row.get("algorithm")
    if not isinstance(algorithm, str):
        return None
    params = row.get("params")
    if not isinstance(params, dict):
        params = {}
    recall_key, recall = recall_value(row)
    target_count = params.get("target_count")
    target_count = int(target_count) if isinstance(target_count, (int, float)) else None
    storage_mode = row.get("storage_mode")
    if not isinstance(storage_mode, str):
        storage_mode = "in_memory"
    cache_state = row.get("cache_state")
    index_bytes = row.get("index_bytes")
    return SelectivityRow(
        workload=workload,
        algorithm=algorithm,
        storage_mode=storage_mode,
        cache_state=cache_state if isinstance(cache_state, str) else None,
        selectivity=number(params.get("selectivity")),
        target_count=target_count,
        recall_key=recall_key,
        recall=recall,
        qps=number(row.get("qps")),
        index_bytes=int(index_bytes) if isinstance(index_bytes, int) else None,
        index_bytes_required=index_bytes_required,
        p95_us=number(row.get("p95_us")),
        p99_us=number(row.get("p99_us")),
        mean_returned=number(row.get("mean_returned")),
        two_hop_invocations=number(row.get("two_hop_invocations")),
        two_hop_nodes_examined=number(row.get("two_hop_nodes_examined")),
    )


def load_rows(paths: list[Path]) -> list[SelectivityRow]:
    rows: list[SelectivityRow] = []
    for path in paths:
        workload = path.stem
        index_bytes_required = False
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    continue
                meta = parsed.get("_meta")
                if isinstance(meta, dict):
                    workload = scoped_workload(meta, path)
                    index_bytes_required = meta.get("index_bytes_required") is True
                    continue
                row = parse_row(
                    parsed, workload, index_bytes_required=index_bytes_required
                )
                if row is not None:
                    rows.append(row)
    return rows


def rows_missing_index_bytes(
    rows: list[SelectivityRow], *, declared_only: bool = False
) -> list[SelectivityRow]:
    return [
        row
        for row in rows
        if row.index_bytes is None and (not declared_only or row.index_bytes_required)
    ]


def require_index_bytes(
    rows: list[SelectivityRow], *, declared_only: bool = False
) -> None:
    missing = rows_missing_index_bytes(rows, declared_only=declared_only)
    if not missing:
        return
    examples = ", ".join(
        f"{row.workload}:{row.algorithm}:{row.storage_mode}:selectivity={row.selectivity}"
        for row in missing[:10]
    )
    suffix = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
    raise SystemExit(f"selectivity rows missing index_bytes: {examples}{suffix}")


def markdown_table(rows: list[SelectivityRow]) -> str:
    headers = [
        "Workload",
        "Algorithm",
        "Storage",
        "Cache",
        "Selectivity",
        "Target",
        "Recall",
        "QPS",
        "Index Bytes",
        "p95 us",
        "p99 us",
        "Returned",
        "2-hop calls",
        "2-hop nodes",
    ]
    aligns = [
        "---",
        "---",
        "---",
        "---",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
        "---:",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            item.workload,
            item.algorithm,
            item.selectivity if item.selectivity is not None else -1.0,
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    row.workload,
                    row.algorithm,
                    row.storage_mode,
                    "" if row.cache_state is None else row.cache_state,
                    fmt_float(row.selectivity, 4),
                    "" if row.target_count is None else str(row.target_count),
                    fmt_float(row.recall, 4),
                    fmt_qps(row.qps),
                    "" if row.index_bytes is None else str(row.index_bytes),
                    fmt_float(row.p95_us),
                    fmt_float(row.p99_us),
                    fmt_float(row.mean_returned),
                    fmt_float(row.two_hop_invocations),
                    fmt_float(row.two_hop_nodes_examined),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def rows_as_json(rows: list[SelectivityRow]) -> list[dict[str, Any]]:
    return [
        {
            "workload": row.workload,
            "algorithm": row.algorithm,
            "storage_mode": row.storage_mode,
            "cache_state": row.cache_state,
            "selectivity": row.selectivity,
            "target_count": row.target_count,
            row.recall_key: row.recall,
            "qps": row.qps,
            "index_bytes": row.index_bytes,
            "index_bytes_required": row.index_bytes_required,
            "p95_us": row.p95_us,
            "p99_us": row.p99_us,
            "mean_returned": row.mean_returned,
            "two_hop_invocations": row.two_hop_invocations,
            "two_hop_nodes_examined": row.two_hop_nodes_examined,
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-index-bytes",
        action="store_true",
        help="Exit non-zero if any selectivity row lacks index_bytes",
    )
    parser.add_argument(
        "--require-declared-index-bytes",
        action="store_true",
        help=(
            "Exit non-zero only for rows under "
            "_meta.index_bytes_required=true that lack index_bytes"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    if args.require_index_bytes:
        require_index_bytes(rows)
    if args.require_declared_index_bytes:
        require_index_bytes(rows, declared_only=True)
    if args.json:
        print(json.dumps(rows_as_json(rows), indent=2) + "\n", end="")
    else:
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
