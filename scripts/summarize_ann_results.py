#!/usr/bin/env python3
"""Summarize ann_benchmark JSONL coverage by dataset, algorithm, and storage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STORAGE_EXPECTATION_GROUPS = {
    ("in_memory", "snapshot_loaded"): (
        "hnsw",
        "nsw",
        "ivf_avq",
        "ivf_rabitq",
        "rp_quant",
        "binary_index",
        "lsh",
        "emg",
        "nsg",
        "fresh_graph",
        "dual_branch",
        "deg",
        "filtered_graph",
        "inplace",
        "pipnn",
        "vamana",
        "finger",
        "sng",
        "symphony_qg",
        "symphony_qg_vr",
        "sq4",
        "sq4u",
        "sq8u",
        "curator",
        "range_filtered",
        "kdtree",
        "balltree",
        "rptree",
        "rp_forest",
        "kmeans_tree",
    ),
    ("in_memory",): (
        "brute",
        "adsampling",
        "hnsw_prt",
        "fresh_graph_churn",
        "inplace_churn",
        "lsm_churn",
        "sparse_mips",
    ),
    ("in_memory", "snapshot_loaded", "file", "mmap"): (
        "ivfpq",
        "ivfpq_rerank",
    ),
    ("segmented_store",): ("store", "store_snapshot"),
}

DISKANN_EXPECTATION_ROWS = (
    ("diskann", "in_memory"),
    ("diskann_file", "file"),
    ("diskann_mmap", "mmap"),
)
ExpectationRow = tuple[str, str]
ExpectationFamily = tuple[frozenset[str], tuple[ExpectationRow, ...]]


def standard_storage_expectation_families() -> tuple[ExpectationFamily, ...]:
    families = []
    for storage_modes, algorithms in STORAGE_EXPECTATION_GROUPS.items():
        for algorithm in algorithms:
            families.append(
                (
                    frozenset({algorithm}),
                    tuple((algorithm, storage_mode) for storage_mode in storage_modes),
                )
            )
    diskann_algorithms = frozenset(
        algorithm for algorithm, _storage_mode in DISKANN_EXPECTATION_ROWS
    )
    families.append((diskann_algorithms, DISKANN_EXPECTATION_ROWS))
    return tuple(families)


STANDARD_STORAGE_EXPECTATION_FAMILIES = standard_storage_expectation_families()


def standard_storage_expectations() -> list[tuple[str, str]]:
    rows = {
        row
        for _observed_algorithms, expected_rows in STANDARD_STORAGE_EXPECTATION_FAMILIES
        for row in expected_rows
    }
    return sorted(rows)


STANDARD_STORAGE_EXPECTATIONS = standard_storage_expectations()

DIAGNOSTIC_KEYS = (
    "avg_visited_nodes",
    "avg_graph_reads",
    "avg_vector_reads",
    "avg_graph_bytes",
    "avg_vector_bytes",
    "avg_retained_candidates",
)


def observed_standard_storage_expectations(
    summaries: dict[tuple[str, str, str], Summary],
) -> dict[str, list[tuple[str, str]]]:
    observed_by_dataset: dict[str, set[str]] = defaultdict(set)
    for (dataset, algorithm, _storage_mode), summary in summaries.items():
        if summary.storage_scope_observed:
            observed_by_dataset[dataset].add(algorithm)

    expected_by_dataset = {}
    families = STANDARD_STORAGE_EXPECTATION_FAMILIES
    for dataset, observed_algorithms in observed_by_dataset.items():
        expected_by_dataset[dataset] = [
            row
            for family_algorithms, expected_rows in families
            if family_algorithms & observed_algorithms
            for row in expected_rows
        ]
    return expected_by_dataset


@dataclass
class Summary:
    rows: int = 0
    best_recall: float = 0.0
    best_qps: float = 0.0
    best_qps_index_bytes: int | None = None
    best_qps_diagnostics: dict[str, float] | None = None
    recall_qps: list[tuple[float, float, int | None]] | None = None
    storage_scope_observed: bool = False

    def add(self, row: dict[str, Any], *, storage_scope_observed: bool = False) -> None:
        self.rows += 1
        self.storage_scope_observed |= storage_scope_observed
        recall = float(row.get("recall_at_10", 0.0))
        qps = float(row.get("qps", 0.0))
        index_bytes = row.get("index_bytes")
        index_bytes = index_bytes if isinstance(index_bytes, int) else None
        self.best_recall = max(self.best_recall, recall)
        if qps >= self.best_qps:
            self.best_qps = qps
            self.best_qps_index_bytes = index_bytes
            diagnostics = {
                key: float(row[key])
                for key in DIAGNOSTIC_KEYS
                if isinstance(row.get(key), int | float)
            }
            self.best_qps_diagnostics = diagnostics or None
        if self.recall_qps is None:
            self.recall_qps = []
        self.recall_qps.append((recall, qps, index_bytes))

    def qps_at_recall(self, recall_floor: float) -> float | None:
        qualifying = [
            qps
            for recall, qps, _index_bytes in self.recall_qps or []
            if recall >= recall_floor
        ]
        return max(qualifying) if qualifying else None

    def index_bytes_at_recall(self, recall_floor: float) -> int | None:
        qualifying = [
            (qps, index_bytes)
            for recall, qps, index_bytes in self.recall_qps or []
            if recall >= recall_floor
        ]
        if not qualifying:
            return None
        _qps, index_bytes = max(qualifying)
        return index_bytes


@dataclass(frozen=True)
class CoverageRow:
    dataset: str
    algorithm: str
    storage_mode: str
    status: str
    rows: int
    best_recall: float | None
    best_qps: float | None
    best_index_bytes: int | None
    qps_at_recall_floor: float | None
    index_bytes_at_recall_floor: int | None
    best_row_diagnostics: dict[str, float] | None


def scoped_dataset_name(meta: dict[str, Any]) -> str | None:
    dataset = meta.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        return None

    name = Path(dataset).name
    scope = []
    train_limit = meta.get("train_limit")
    query_limit = meta.get("query_limit")
    if train_limit is not None:
        scope.append(f"train={train_limit}")
    if query_limit is not None:
        scope.append(f"queries={query_limit}")
    if scope:
        name = f"{name}[{','.join(scope)}]"
    return name


def has_storage_expectation_scope(meta: dict[str, Any]) -> bool:
    storage_scope_keys = ("indexed_vectors", "queries", "train_limit", "query_limit")
    return any(key in meta for key in storage_scope_keys)


def row_dataset(row: dict[str, Any], current_dataset: str | None, path: Path) -> str:
    dataset = row.get("dataset") or current_dataset
    if isinstance(dataset, str) and dataset:
        return Path(dataset).name
    return path.stem


def load_summaries(
    paths: list[Path], *, current_schema_only: bool = False
) -> dict[tuple[str, str, str], Summary]:
    summaries: dict[tuple[str, str, str], Summary] = defaultdict(Summary)
    for path in paths:
        current_dataset: str | None = None
        storage_expectation_scope = False
        seen_meta = False
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "_meta" in row:
                    meta = row["_meta"]
                    if isinstance(meta, dict):
                        current_dataset = scoped_dataset_name(meta)
                        storage_expectation_scope = has_storage_expectation_scope(meta)
                        seen_meta = True
                    continue
                if current_schema_only and not seen_meta:
                    continue
                algorithm = row.get("algorithm")
                if not isinstance(algorithm, str):
                    continue
                storage_mode = row.get("storage_mode")
                if not isinstance(storage_mode, str):
                    storage_mode = "in_memory"
                dataset = row_dataset(row, current_dataset, path)
                summaries[(dataset, algorithm, storage_mode)].add(
                    row,
                    storage_scope_observed=storage_expectation_scope,
                )
    return dict(summaries)


def coverage_rows(
    summaries: dict[tuple[str, str, str], Summary],
    expected: list[tuple[str, str]] | None = None,
    expected_by_dataset: dict[str, list[tuple[str, str]]] | None = None,
    datasets: list[str] | None = None,
    recall_floor: float = 0.95,
    only_datasets: set[str] | None = None,
    missing_only: bool = False,
) -> list[CoverageRow]:
    expected = expected or []
    expected_by_dataset = expected_by_dataset or {}
    dataset_names = sorted(
        datasets
        or ({dataset for dataset, _, _ in summaries} | set(expected_by_dataset))
    )
    if only_datasets is not None:
        dataset_names = [
            dataset for dataset in dataset_names if dataset in only_datasets
        ]
    keys = set(summaries)
    if expected:
        for dataset in dataset_names:
            for algorithm, storage_mode in expected:
                keys.add((dataset, algorithm, storage_mode))
    for dataset, dataset_expected in expected_by_dataset.items():
        if dataset in dataset_names:
            for algorithm, storage_mode in dataset_expected:
                keys.add((dataset, algorithm, storage_mode))

    rows = []
    for dataset, algorithm, storage_mode in sorted(keys):
        if only_datasets is not None and dataset not in only_datasets:
            continue
        summary = summaries.get((dataset, algorithm, storage_mode))
        if missing_only and summary:
            continue
        rows.append(
            CoverageRow(
                dataset=dataset,
                algorithm=algorithm,
                storage_mode=storage_mode,
                status="measured" if summary else "missing",
                rows=summary.rows if summary else 0,
                best_recall=summary.best_recall if summary else None,
                best_qps=summary.best_qps if summary else None,
                best_index_bytes=summary.best_qps_index_bytes if summary else None,
                qps_at_recall_floor=(
                    summary.qps_at_recall(recall_floor) if summary else None
                ),
                index_bytes_at_recall_floor=(
                    summary.index_bytes_at_recall(recall_floor) if summary else None
                ),
                best_row_diagnostics=summary.best_qps_diagnostics if summary else None,
            )
        )
    return rows


def markdown_table(rows: list[CoverageRow], recall_floor: float = 0.95) -> str:
    lines = [
        "| Dataset | Algorithm | Storage | Status | Rows | Best Recall@10 | Best QPS | "
        f"Best Index Bytes | Best QPS @ R>={recall_floor:.2f} | "
        f"Index Bytes @ R>={recall_floor:.2f} |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        recall = "-" if row.best_recall is None else f"{row.best_recall:.4f}"
        qps = "-" if row.best_qps is None else f"{row.best_qps:.1f}"
        index_bytes = "-" if row.best_index_bytes is None else str(row.best_index_bytes)
        floor_qps = (
            "-" if row.qps_at_recall_floor is None else f"{row.qps_at_recall_floor:.1f}"
        )
        floor_index_bytes = (
            "-"
            if row.index_bytes_at_recall_floor is None
            else str(row.index_bytes_at_recall_floor)
        )
        lines.append(
            f"| {row.dataset} | {row.algorithm} | {row.storage_mode} | "
            f"{row.status} | {row.rows} | {recall} | {qps} | {index_bytes} | "
            f"{floor_qps} | {floor_index_bytes} |"
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
        "--expect-standard-storage",
        action="store_true",
        help="Add the standard storage-coverage expectation matrix",
    )
    parser.add_argument(
        "--expect-observed-standard-storage",
        action="store_true",
        help=(
            "Add standard storage expectations only for algorithm families "
            "already observed in each dataset"
        ),
    )
    parser.add_argument(
        "--current-schema-only",
        action="store_true",
        help=(
            "Ignore legacy JSONL files or rows that do not follow a current _meta line"
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset name to use for expected rows when no measured row exists",
    )
    parser.add_argument(
        "--only-dataset",
        action="append",
        default=[],
        help="Restrict output to one dataset name. May be repeated.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Emit only expected rows that are missing",
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=0.95,
        help="Recall@10 floor used for thresholded QPS reporting",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(
        args.paths,
        current_schema_only=args.current_schema_only,
    )
    expected = list(args.expect)
    if args.expect_standard_storage:
        expected.extend(STANDARD_STORAGE_EXPECTATIONS)
    expected_by_dataset = (
        observed_standard_storage_expectations(summaries)
        if args.expect_observed_standard_storage
        else None
    )
    only_datasets = set(args.only_dataset) if args.only_dataset else None
    rows = coverage_rows(
        summaries,
        expected,
        expected_by_dataset,
        args.dataset,
        args.recall_floor,
        only_datasets=only_datasets,
        missing_only=args.missing_only,
    )
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
                        "best_index_bytes": row.best_index_bytes,
                        "best_row_diagnostics": row.best_row_diagnostics,
                        "qps_at_recall_floor": row.qps_at_recall_floor,
                        "index_bytes_at_recall_floor": row.index_bytes_at_recall_floor,
                    }
                    for row in rows
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(markdown_table(rows, args.recall_floor))


if __name__ == "__main__":
    main()
