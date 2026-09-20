from __future__ import annotations

import importlib.util
import itertools
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("matplotlib")


def load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts/plot_pareto.py"
    spec = importlib.util.spec_from_file_location("plot_pareto", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def point(ef: int, recall: float, qps: float) -> dict:
    return {"ef": ef, "recall_mean": recall, "qps_mean": qps}


def test_pareto_frontier_removes_dominated_points_and_ties() -> None:
    script = load_script()
    points = [
        point(10, 0.80, 500),
        point(20, 0.80, 400),
        point(30, 0.90, 500),
        point(40, 0.90, 200),
        point(50, 0.95, 100),
        point(60, 0.90, 500),
    ]

    expected = [
        (30, 0.90, 500),
        (50, 0.95, 100),
    ]
    for ordering in itertools.permutations(points):
        frontier = script.pareto_frontier(list(ordering))
        assert [
            (item["ef"], item["recall_mean"], item["qps_mean"]) for item in frontier
        ] == expected


def test_fastest_at_recall_rejects_subthreshold_points() -> None:
    script = load_script()

    assert (
        script.fastest_at_recall([point(10, 0.89, 1_000), point(20, 0.91, 100)], 0.90)
        == 100
    )
    assert math.isnan(script.fastest_at_recall([point(10, 0.89, 1_000)], 0.90))
