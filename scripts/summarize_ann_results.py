#!/usr/bin/env python3
"""Summarize ann_benchmark JSONL coverage by dataset, algorithm, and storage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

STORAGE_EXPECTATION_GROUPS = {
    ("in_memory", "snapshot_loaded"): (
        "hnsw",
        "nsw",
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
        "lsm_churn",
    ),
    ("in_memory",): (
        "brute",
        "adsampling",
        "hnsw_prt",
        "fresh_graph_churn",
        "inplace_churn",
        "sparse_mips",
    ),
    ("in_memory", "snapshot_loaded", "file", "mmap"): (
        "ivfpq",
        "ivfpq_rerank",
        "ivf_avq",
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
    "avg_probed_lists",
    "avg_scanned_vectors",
    "avg_partition_reads",
    "avg_partition_bytes",
    "avg_graph_reads",
    "avg_code_reads",
    "avg_vector_reads",
    "avg_page_reads",
    "avg_graph_bytes",
    "avg_code_bytes",
    "avg_vector_bytes",
    "avg_page_bytes",
    "avg_retained_candidates",
    "active_count",
    "update_time_s",
    "update_qps",
    "tombstone_ratio",
    "free_slot_ratio",
    "compactions",
    "levels",
    "tombstones",
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
    best_recall_params: dict[str, Any] | None = None
    best_recall_index_bytes: int | None = None
    best_recall_index_bytes_kind: str | None = None
    best_qps: float = 0.0
    best_qps_params: dict[str, Any] | None = None
    best_qps_index_bytes: int | None = None
    best_qps_index_bytes_kind: str | None = None
    best_qps_diagnostics: dict[str, float] | None = None
    recall_qps: (
        list[tuple[float, float, int | None, str | None, dict[str, Any] | None]] | None
    ) = None
    storage_scope_observed: bool = False
    index_bytes_required: bool = False
    missing_index_bytes_rows: int = 0
    required_missing_index_bytes_rows: int = 0

    def add(
        self,
        row: dict[str, Any],
        *,
        storage_scope_observed: bool = False,
        index_bytes_required: bool = False,
    ) -> None:
        self.rows += 1
        self.storage_scope_observed |= storage_scope_observed
        self.index_bytes_required |= index_bytes_required
        recall = float(row.get("recall_at_10", 0.0))
        qps = float(row.get("qps", 0.0))
        index_bytes = row.get("index_bytes")
        index_bytes = index_bytes if isinstance(index_bytes, int) else None
        index_bytes_kind = row.get("index_bytes_kind")
        index_bytes_kind = (
            index_bytes_kind if isinstance(index_bytes_kind, str) else None
        )
        if index_bytes is None:
            self.missing_index_bytes_rows += 1
            if index_bytes_required:
                self.required_missing_index_bytes_rows += 1
        params = row.get("params")
        params = dict(params) if isinstance(params, dict) else None
        if recall >= self.best_recall:
            self.best_recall = recall
            self.best_recall_params = params
            self.best_recall_index_bytes = index_bytes
            self.best_recall_index_bytes_kind = index_bytes_kind
        if qps >= self.best_qps:
            self.best_qps = qps
            self.best_qps_params = params
            self.best_qps_index_bytes = index_bytes
            self.best_qps_index_bytes_kind = index_bytes_kind
            diagnostics = {
                key: float(row[key])
                for key in DIAGNOSTIC_KEYS
                if isinstance(row.get(key), (int, float))
            }
            self.best_qps_diagnostics = diagnostics or None
        if self.recall_qps is None:
            self.recall_qps = []
        self.recall_qps.append((recall, qps, index_bytes, index_bytes_kind, params))

    def best_at_recall(
        self,
        recall_floor: float,
    ) -> tuple[float, int | None, str | None, dict[str, Any] | None] | None:
        qualifying = [
            (qps, index_bytes, index_bytes_kind, params)
            for recall, qps, index_bytes, index_bytes_kind, params in self.recall_qps
            or []
            if recall >= recall_floor
        ]
        if not qualifying:
            return None
        return max(qualifying, key=lambda item: item[0])

    def qps_at_recall(self, recall_floor: float) -> float | None:
        best = self.best_at_recall(recall_floor)
        return None if best is None else best[0]

    def index_bytes_at_recall(self, recall_floor: float) -> int | None:
        best = self.best_at_recall(recall_floor)
        return None if best is None else best[1]

    def index_bytes_kind_at_recall(self, recall_floor: float) -> str | None:
        best = self.best_at_recall(recall_floor)
        return None if best is None else best[2]

    def params_at_recall(self, recall_floor: float) -> dict[str, Any] | None:
        best = self.best_at_recall(recall_floor)
        return None if best is None else best[3]


@dataclass(frozen=True)
class DatasetProfile:
    path: str
    metric: str | None
    train: int | None
    dim: int | None
    pair_distance_p50: float | None
    nearest_distance_p50: float | None
    top2_gap_p50: float | None
    lid_p50: float | None
    contrast_p50: float | None
    hub_gini: float | None
    coarse_gini: float | None
    split_kinds: str | None


@dataclass(frozen=True)
class CoverageRow:
    dataset: str
    dataset_profile: str | None
    profile_metric: str | None
    profile_train: int | None
    profile_dim: int | None
    profile_pair_distance_p50: float | None
    profile_nearest_distance_p50: float | None
    profile_top2_gap_p50: float | None
    profile_lid_p50: float | None
    profile_contrast_p50: float | None
    profile_hub_gini: float | None
    profile_coarse_gini: float | None
    profile_split_kinds: str | None
    algorithm: str
    storage_mode: str
    status: str
    rows: int
    best_recall: float | None
    best_recall_params: dict[str, Any] | None
    best_recall_index_bytes: int | None
    best_recall_index_bytes_kind: str | None
    best_qps: float | None
    best_params: dict[str, Any] | None
    best_index_bytes: int | None
    best_index_bytes_kind: str | None
    qps_at_recall_floor: float | None
    params_at_recall_floor: dict[str, Any] | None
    index_bytes_at_recall_floor: int | None
    index_bytes_kind_at_recall_floor: str | None
    best_row_diagnostics: dict[str, float] | None
    storage_scope_observed: bool
    index_bytes_required: bool
    missing_index_bytes_rows: int
    required_missing_index_bytes_rows: int


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


def is_current_ann_result_meta(meta: dict[str, Any]) -> bool:
    return isinstance(meta.get("dataset"), str) and meta.get("result_schema") == 2


def has_storage_expectation_scope(meta: dict[str, Any]) -> bool:
    storage_scope_keys = ("indexed_vectors", "queries", "train_limit", "query_limit")
    return any(key in meta for key in storage_scope_keys)


def row_dataset(row: dict[str, Any], current_dataset: str | None, path: Path) -> str:
    dataset = row.get("dataset") or current_dataset
    if isinstance(dataset, str) and dataset:
        return Path(dataset).name
    return path.stem


def profile_dataset_name(profile: dict[str, Any], path: Path) -> str:
    dataset = profile.get("dataset")
    if isinstance(dataset, str) and dataset:
        return Path(dataset).name
    return path.stem


def nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def optional_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def split_kinds(profile: dict[str, Any]) -> str | None:
    splits = profile.get("query_splits")
    if not isinstance(splits, list):
        return None
    kinds = [
        str(split.get("kind"))
        for split in splits
        if isinstance(split, dict) and split.get("kind")
    ]
    return ",".join(kinds) if kinds else None


def dataset_profile(path: Path, profile: dict[str, Any]) -> DatasetProfile:
    metric = profile.get("metric")
    return DatasetProfile(
        path=str(path),
        metric=metric if isinstance(metric, str) else None,
        train=optional_int(nested(profile, ("shape", "train"))),
        dim=optional_int(nested(profile, ("shape", "dim"))),
        pair_distance_p50=optional_float(
            nested(profile, ("pair_distance_sample", "p50"))
        ),
        nearest_distance_p50=optional_float(
            nested(profile, ("query_neighbors", "nearest_distance", "p50"))
        ),
        top2_gap_p50=optional_float(
            nested(profile, ("query_neighbors", "top2_gap", "p50"))
        ),
        lid_p50=optional_float(nested(profile, ("query_neighbors", "lid_mle", "p50"))),
        contrast_p50=optional_float(
            nested(profile, ("sampled_relative_contrast", "p50"))
        ),
        hub_gini=optional_float(nested(profile, ("hubness", "gini"))),
        coarse_gini=optional_float(
            nested(profile, ("coarse_partition_imbalance", "count_gini"))
        ),
        split_kinds=split_kinds(profile),
    )


def load_dataset_profiles(profile_dirs: list[Path]) -> dict[str, DatasetProfile]:
    profiles: dict[str, DatasetProfile] = {}
    for profile_dir in profile_dirs:
        for path in sorted(profile_dir.glob("*.json")):
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path} is not valid JSON") from exc
            if not isinstance(profile, dict):
                raise SystemExit(f"{path} does not contain a JSON object")
            dataset = profile_dataset_name(profile, path)
            existing = profiles.get(dataset)
            if existing is not None and existing.path != str(path):
                raise SystemExit(
                    f"multiple profiles found for dataset {dataset}: "
                    f"{existing.path} and {path}"
                )
            profiles[dataset] = dataset_profile(path, profile)
    return profiles


def load_summaries(
    paths: list[Path],
    *,
    current_schema_only: bool = False,
    footprint_contract_only: bool = False,
) -> dict[tuple[str, str, str], Summary]:
    summaries: dict[tuple[str, str, str], Summary] = defaultdict(Summary)
    for path in paths:
        current_dataset: str | None = None
        storage_expectation_scope = False
        index_bytes_required = False
        active_current_schema = not current_schema_only
        active_footprint_contract = not footprint_contract_only
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "_meta" in row:
                    meta = row["_meta"]
                    if isinstance(meta, dict):
                        if current_schema_only and not is_current_ann_result_meta(meta):
                            current_dataset = None
                            storage_expectation_scope = False
                            index_bytes_required = False
                            active_current_schema = False
                            active_footprint_contract = False
                            continue
                        current_dataset = scoped_dataset_name(meta)
                        storage_expectation_scope = has_storage_expectation_scope(meta)
                        index_bytes_required = meta.get("index_bytes_required") is True
                        active_current_schema = True
                        active_footprint_contract = (
                            not footprint_contract_only or index_bytes_required
                        )
                    continue
                if current_schema_only and not active_current_schema:
                    continue
                if footprint_contract_only and not active_footprint_contract:
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
                    index_bytes_required=index_bytes_required,
                )
    return dict(summaries)


def coverage_rows(
    summaries: dict[tuple[str, str, str], Summary],
    expected: list[tuple[str, str]] | None = None,
    expected_by_dataset: dict[str, list[tuple[str, str]]] | None = None,
    datasets: list[str] | None = None,
    dataset_profiles: dict[str, DatasetProfile] | None = None,
    recall_floor: float = 0.95,
    only_datasets: set[str] | None = None,
    missing_only: bool = False,
    recall_gap_only: bool = False,
) -> list[CoverageRow]:
    expected = expected or []
    expected_by_dataset = expected_by_dataset or {}
    dataset_profiles = dataset_profiles or {}
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
        profile = dataset_profiles.get(dataset)
        if missing_only and summary:
            continue
        if recall_gap_only and (
            summary is None
            or not summary.storage_scope_observed
            or summary.qps_at_recall(recall_floor) is not None
        ):
            continue
        rows.append(
            CoverageRow(
                dataset=dataset,
                dataset_profile=profile.path if profile else None,
                profile_metric=profile.metric if profile else None,
                profile_train=profile.train if profile else None,
                profile_dim=profile.dim if profile else None,
                profile_pair_distance_p50=(
                    profile.pair_distance_p50 if profile else None
                ),
                profile_nearest_distance_p50=(
                    profile.nearest_distance_p50 if profile else None
                ),
                profile_top2_gap_p50=profile.top2_gap_p50 if profile else None,
                profile_lid_p50=profile.lid_p50 if profile else None,
                profile_contrast_p50=profile.contrast_p50 if profile else None,
                profile_hub_gini=profile.hub_gini if profile else None,
                profile_coarse_gini=profile.coarse_gini if profile else None,
                profile_split_kinds=profile.split_kinds if profile else None,
                algorithm=algorithm,
                storage_mode=storage_mode,
                status="measured" if summary else "missing",
                rows=summary.rows if summary else 0,
                best_recall=summary.best_recall if summary else None,
                best_recall_params=summary.best_recall_params if summary else None,
                best_recall_index_bytes=(
                    summary.best_recall_index_bytes if summary else None
                ),
                best_recall_index_bytes_kind=(
                    summary.best_recall_index_bytes_kind if summary else None
                ),
                best_qps=summary.best_qps if summary else None,
                best_params=summary.best_qps_params if summary else None,
                best_index_bytes=summary.best_qps_index_bytes if summary else None,
                best_index_bytes_kind=(
                    summary.best_qps_index_bytes_kind if summary else None
                ),
                qps_at_recall_floor=(
                    summary.qps_at_recall(recall_floor) if summary else None
                ),
                params_at_recall_floor=(
                    summary.params_at_recall(recall_floor) if summary else None
                ),
                index_bytes_at_recall_floor=(
                    summary.index_bytes_at_recall(recall_floor) if summary else None
                ),
                index_bytes_kind_at_recall_floor=(
                    summary.index_bytes_kind_at_recall(recall_floor)
                    if summary
                    else None
                ),
                best_row_diagnostics=summary.best_qps_diagnostics if summary else None,
                storage_scope_observed=(
                    summary.storage_scope_observed if summary else False
                ),
                index_bytes_required=summary.index_bytes_required if summary else False,
                missing_index_bytes_rows=(
                    summary.missing_index_bytes_rows if summary else 0
                ),
                required_missing_index_bytes_rows=(
                    summary.required_missing_index_bytes_rows if summary else 0
                ),
            )
        )
    return rows


def format_params(params: dict[str, Any] | None) -> str:
    if params is None:
        return "-"
    return f"`{json.dumps(params, sort_keys=True, separators=(',', ':'))}`"


def dataset_base_and_train_scope(dataset: str) -> tuple[str, int | None]:
    if "[" not in dataset or not dataset.endswith("]"):
        return dataset, None
    base, scope = dataset[:-1].split("[", 1)
    train = None
    for part in scope.split(","):
        key, sep, value = part.partition("=")
        if key == "train" and sep:
            try:
                train = int(value)
            except ValueError:
                train = None
    return base, train


def train_scope_dominates(candidate: int | None, gap: int | None) -> bool:
    if candidate is None:
        return True
    if gap is None:
        return False
    return candidate >= gap


def recall_gap_row(row: CoverageRow) -> bool:
    return (
        row.status == "measured"
        and row.storage_scope_observed
        and row.qps_at_recall_floor is None
    )


def suppress_dominated_recall_gaps(rows: list[CoverageRow]) -> list[CoverageRow]:
    qualifying_scopes: dict[tuple[str, str, str], list[int | None]] = defaultdict(list)
    for row in rows:
        if row.status != "measured" or row.qps_at_recall_floor is None:
            continue
        base, train_scope = dataset_base_and_train_scope(row.dataset)
        qualifying_scopes[(base, row.algorithm, row.storage_mode)].append(train_scope)

    filtered = []
    for row in rows:
        if not recall_gap_row(row):
            filtered.append(row)
            continue
        base, gap_train_scope = dataset_base_and_train_scope(row.dataset)
        candidates = qualifying_scopes.get((base, row.algorithm, row.storage_mode), [])
        if any(
            train_scope_dominates(candidate, gap_train_scope)
            for candidate in candidates
        ):
            continue
        filtered.append(row)
    return filtered


def format_optional_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def format_optional_float(value: float | None) -> str:
    return "-" if value is None else f"{value:.3g}"


def rows_missing_index_bytes(
    rows: list[CoverageRow], *, declared_only: bool = False
) -> list[CoverageRow]:
    return [
        row
        for row in rows
        if row.status == "measured"
        and (
            row.required_missing_index_bytes_rows > 0
            if declared_only
            else row.missing_index_bytes_rows > 0
        )
    ]


def format_row_key(row: CoverageRow) -> str:
    return f"{row.dataset}:{row.algorithm}:{row.storage_mode}"


def require_index_bytes(
    rows: list[CoverageRow], *, declared_only: bool = False
) -> None:
    missing = rows_missing_index_bytes(rows, declared_only=declared_only)
    if not missing:
        return

    limit = 20
    shown = ", ".join(format_row_key(row) for row in missing[:limit])
    hidden = len(missing) - limit
    suffix = "" if hidden <= 0 else f", ... ({hidden} more)"
    print(
        f"measured rows missing index_bytes: {shown}{suffix}",
        file=sys.stderr,
    )
    raise SystemExit(1)


def markdown_table(rows: list[CoverageRow], recall_floor: float = 0.95) -> str:
    lines = [
        "| Dataset | Profile | Metric | Train | Dim | Pair p50 | NN p50 | "
        "Top-2 Gap p50 | LID p50 | Contrast p50 | Hub Gini | Coarse Gini | "
        "Splits | Algorithm | Storage | Status | Rows | "
        "Best Recall@10 | Params @ Best Recall | Index Bytes @ Best Recall | "
        "Index Bytes Kind @ Best Recall | Best QPS | Best Params | "
        "Best Index Bytes | Best Index Bytes Kind | "
        f"Best QPS @ R>={recall_floor:.2f} | "
        f"Params @ R>={recall_floor:.2f} | "
        f"Index Bytes @ R>={recall_floor:.2f} | "
        f"Index Bytes Kind @ R>={recall_floor:.2f} |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | "
        "---: | --- | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        profile = "-" if row.dataset_profile is None else row.dataset_profile
        profile_metric = row.profile_metric or "-"
        profile_split_kinds = row.profile_split_kinds or "-"
        recall = "-" if row.best_recall is None else f"{row.best_recall:.4f}"
        recall_params = format_params(row.best_recall_params)
        recall_index_bytes = (
            "-"
            if row.best_recall_index_bytes is None
            else str(row.best_recall_index_bytes)
        )
        recall_index_bytes_kind = row.best_recall_index_bytes_kind or "-"
        qps = "-" if row.best_qps is None else f"{row.best_qps:.1f}"
        best_params = format_params(row.best_params)
        index_bytes = "-" if row.best_index_bytes is None else str(row.best_index_bytes)
        index_bytes_kind = row.best_index_bytes_kind or "-"
        floor_qps = (
            "-" if row.qps_at_recall_floor is None else f"{row.qps_at_recall_floor:.1f}"
        )
        floor_params = format_params(row.params_at_recall_floor)
        floor_index_bytes = (
            "-"
            if row.index_bytes_at_recall_floor is None
            else str(row.index_bytes_at_recall_floor)
        )
        floor_index_bytes_kind = row.index_bytes_kind_at_recall_floor or "-"
        lines.append(
            f"| {row.dataset} | {profile} | {profile_metric} | "
            f"{format_optional_int(row.profile_train)} | "
            f"{format_optional_int(row.profile_dim)} | "
            f"{format_optional_float(row.profile_pair_distance_p50)} | "
            f"{format_optional_float(row.profile_nearest_distance_p50)} | "
            f"{format_optional_float(row.profile_top2_gap_p50)} | "
            f"{format_optional_float(row.profile_lid_p50)} | "
            f"{format_optional_float(row.profile_contrast_p50)} | "
            f"{format_optional_float(row.profile_hub_gini)} | "
            f"{format_optional_float(row.profile_coarse_gini)} | "
            f"{profile_split_kinds} | {row.algorithm} | {row.storage_mode} | "
            f"{row.status} | {row.rows} | {recall} | {recall_params} | "
            f"{recall_index_bytes} | {recall_index_bytes_kind} | {qps} | "
            f"{best_params} | {index_bytes} | {index_bytes_kind} | "
            f"{floor_qps} | {floor_params} | {floor_index_bytes} | "
            f"{floor_index_bytes_kind} |"
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
        "--footprint-contract-only",
        action="store_true",
        help=(
            "Ignore measured rows whose active _meta does not declare "
            "index_bytes_required=true"
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
        "--recall-gap-only",
        action="store_true",
        help=(
            "Emit only scoped measured rows that have no QPS at the recall floor. "
            "This identifies fixed-recall sweep gaps without flagging exploratory rows."
        ),
    )
    parser.add_argument(
        "--suppress-dominated-recall-gaps",
        action="store_true",
        help=(
            "When used with --recall-gap-only, hide a scoped gap if the same "
            "base dataset, algorithm, and storage mode has another scoped row "
            "at the recall floor with at least the same train limit."
        ),
    )
    parser.add_argument(
        "--profile-dir",
        action="append",
        default=[],
        type=Path,
        help=(
            "Directory containing profile_ann_dataset JSON outputs. Rows whose "
            "dataset label exactly matches a profile dataset include its path."
        ),
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=0.95,
        help="Recall@10 floor used for thresholded QPS reporting",
    )
    parser.add_argument(
        "--require-index-bytes",
        action="store_true",
        help="Exit non-zero if any measured summary row lacks index_bytes",
    )
    parser.add_argument(
        "--require-declared-index-bytes",
        action="store_true",
        help=(
            "Exit non-zero only for measured rows under "
            "_meta.index_bytes_required=true that lack index_bytes"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(
        args.paths,
        current_schema_only=args.current_schema_only,
        footprint_contract_only=args.footprint_contract_only,
    )
    expected = list(args.expect)
    if args.expect_standard_storage:
        expected.extend(STANDARD_STORAGE_EXPECTATIONS)
    expected_by_dataset = (
        observed_standard_storage_expectations(summaries)
        if args.expect_observed_standard_storage
        else None
    )
    dataset_profiles = load_dataset_profiles(args.profile_dir)
    only_datasets = set(args.only_dataset) if args.only_dataset else None
    rows = coverage_rows(
        summaries,
        expected,
        expected_by_dataset,
        args.dataset,
        dataset_profiles,
        args.recall_floor,
        only_datasets=only_datasets,
        missing_only=args.missing_only,
        recall_gap_only=(
            args.recall_gap_only and not args.suppress_dominated_recall_gaps
        ),
    )
    if args.suppress_dominated_recall_gaps:
        rows = suppress_dominated_recall_gaps(rows)
        if args.recall_gap_only:
            rows = [row for row in rows if recall_gap_row(row)]
    if args.require_index_bytes:
        require_index_bytes(rows)
    if args.require_declared_index_bytes:
        require_index_bytes(rows, declared_only=True)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "dataset": row.dataset,
                        "dataset_profile": row.dataset_profile,
                        "profile_metric": row.profile_metric,
                        "profile_train": row.profile_train,
                        "profile_dim": row.profile_dim,
                        "profile_pair_distance_p50": row.profile_pair_distance_p50,
                        "profile_nearest_distance_p50": (
                            row.profile_nearest_distance_p50
                        ),
                        "profile_top2_gap_p50": row.profile_top2_gap_p50,
                        "profile_lid_p50": row.profile_lid_p50,
                        "profile_contrast_p50": row.profile_contrast_p50,
                        "profile_hub_gini": row.profile_hub_gini,
                        "profile_coarse_gini": row.profile_coarse_gini,
                        "profile_split_kinds": row.profile_split_kinds,
                        "algorithm": row.algorithm,
                        "storage_mode": row.storage_mode,
                        "status": row.status,
                        "rows": row.rows,
                        "best_recall_at_10": row.best_recall,
                        "best_recall_params": row.best_recall_params,
                        "best_recall_index_bytes": row.best_recall_index_bytes,
                        "best_recall_index_bytes_kind": (
                            row.best_recall_index_bytes_kind
                        ),
                        "best_qps": row.best_qps,
                        "best_params": row.best_params,
                        "best_index_bytes": row.best_index_bytes,
                        "best_index_bytes_kind": row.best_index_bytes_kind,
                        "best_row_diagnostics": row.best_row_diagnostics,
                        "qps_at_recall_floor": row.qps_at_recall_floor,
                        "params_at_recall_floor": row.params_at_recall_floor,
                        "index_bytes_at_recall_floor": row.index_bytes_at_recall_floor,
                        "index_bytes_kind_at_recall_floor": (
                            row.index_bytes_kind_at_recall_floor
                        ),
                        "index_bytes_required": row.index_bytes_required,
                        "missing_index_bytes_rows": row.missing_index_bytes_rows,
                        "required_missing_index_bytes_rows": (
                            row.required_missing_index_bytes_rows
                        ),
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
