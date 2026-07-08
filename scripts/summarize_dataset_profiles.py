#!/usr/bin/env python3
"""Summarize profile_ann_dataset JSON outputs as a compact table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def fmt_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.3g}"


def fmt_int(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{int(value):,}"


def split_kinds(profile: dict[str, Any]) -> str:
    splits = profile.get("query_splits")
    if not isinstance(splits, list):
        return ""
    kinds = [
        str(split.get("kind"))
        for split in splits
        if isinstance(split, dict) and split.get("kind")
    ]
    return ",".join(kinds)


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise SystemExit(f"{path} does not contain a JSON object")
    profile["_path"] = str(path)
    return profile


def row(profile: dict[str, Any]) -> list[str]:
    dataset = Path(str(profile.get("dataset") or profile.get("_path"))).name
    return [
        dataset,
        str(profile.get("metric", "")),
        fmt_int(nested(profile, ("shape", "train"))),
        fmt_int(nested(profile, ("shape", "dim"))),
        fmt_float(nested(profile, ("pair_distance_sample", "p50"))),
        fmt_float(nested(profile, ("query_neighbors", "nearest_distance", "p50"))),
        fmt_float(nested(profile, ("query_neighbors", "top2_gap", "p50"))),
        fmt_float(nested(profile, ("query_neighbors", "lid_mle", "p50"))),
        fmt_float(nested(profile, ("sampled_relative_contrast", "p50"))),
        fmt_float(nested(profile, ("hubness", "gini"))),
        fmt_float(nested(profile, ("coarse_partition_imbalance", "count_gini"))),
        split_kinds(profile),
    ]


def markdown_table(profiles: list[dict[str, Any]]) -> str:
    headers = [
        "Dataset",
        "Metric",
        "Train",
        "Dim",
        "Pair p50",
        "NN p50",
        "Top-2 gap p50",
        "LID p50",
        "Contrast p50",
        "Hub Gini",
        "Coarse Gini",
        "Splits",
    ]
    aligns = [
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
        "---",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for profile in sorted(profiles, key=lambda item: str(item.get("dataset", ""))):
        lines.append("| " + " | ".join(row(profile)) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = [load_profile(path) for path in args.profiles]
    if args.json:
        print(
            json.dumps([row(profile) for profile in profiles], indent=2) + "\n", end=""
        )
    else:
        print(markdown_table(profiles))


if __name__ == "__main__":
    main()
