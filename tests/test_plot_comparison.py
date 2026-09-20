from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("matplotlib")


def load_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts/plot_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_comparison", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frontier_is_order_independent_and_removes_dominated_ties() -> None:
    script = load_script()
    points = [(0.9, 100), (0.9, 200), (0.8, 200), (1.0, 50), (0.9, 200)]
    for ordering in itertools.permutations(points):
        assert script.pareto_frontier(ordering) == [(0.9, 200), (1.0, 50)]


def test_frontier_keeps_real_tradeoffs() -> None:
    script = load_script()
    assert script.pareto_frontier([]) == []
    assert script.pareto_frontier([(0.7, 400), (0.9, 200), (1.0, 50)]) == [
        (0.7, 400),
        (0.9, 200),
        (1.0, 50),
    ]


@pytest.mark.parametrize(
    "qps,recall",
    [(0, 0.9), (-1, 0.9), (float("inf"), 0.9), (1, float("nan")), (1, 1.1)],
)
def test_load_results_rejects_invalid_measurements(tmp_path, qps, recall) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"algorithm": "hnsw", "qps": qps, "recall_at_10": recall})
    )
    with pytest.raises(ValueError, match="Invalid recall/QPS"):
        load_script().load_results([path])


def test_load_results_rejects_mixed_machines_but_allows_repeats(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    first = {"dataset": "glove", "cpu": "machine-a", "repeat": 0}
    second = {**first, "repeat": 1}
    path.write_text("\n".join(json.dumps({"_meta": row}) for row in [first, second]))
    load_script().load_results([path])
    second["cpu"] = "machine-b"
    path.write_text("\n".join(json.dumps({"_meta": row}) for row in [first, second]))
    with pytest.raises(ValueError, match="Incompatible benchmark metadata"):
        load_script().load_results([path])


def test_plot_writes_raster_and_vector_outputs(tmp_path) -> None:
    load_script().plot_one_dataset(
        "fixture", {"hnsw": [(0.9, 200), (1.0, 50)]}, tmp_path
    )
    assert (
        (tmp_path / "algorithm_comparison_fixture.png")
        .read_bytes()
        .startswith(b"\x89PNG")
    )
    assert "<svg" in (tmp_path / "algorithm_comparison_fixture.svg").read_text()


def test_series_separates_cache_state_and_actual_result_depth() -> None:
    script = load_script()
    row = {"algorithm": "hnsw", "cache_state": "warm_after_build", "search_k": 100}
    assert script.series_key(row) == "hnsw:warm_after_build, k=100"
    assert script.series_key(row) != script.series_key({**row, "search_k": 10})
    assert script.series_key(row) != script.series_key({**row, "cache_state": "cold"})


def test_load_results_groups_current_schema_by_scoped_dataset(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":500}}\n'
        '{"algorithm":"ivfpq","storage_mode":"file","recall_at_10":0.95,"qps":2500}\n'
        '{"algorithm":"ivfpq","storage_mode":"mmap","recall_at_10":0.96,"qps":2700}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","train_limit":50000,"query_limit":1000}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.97,"qps":11000}\n',
        encoding="utf-8",
    )

    by_dataset = script.load_results([path])

    assert by_dataset == {
        "glove-25-angular[queries=500]": {
            "ivfpq:file": [(0.95, 2500.0)],
            "ivfpq:mmap": [(0.96, 2700.0)],
        },
        "glove-25-angular[train=50000,queries=1000]": {"hnsw": [(0.97, 11000.0)]},
    }
