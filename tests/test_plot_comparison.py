from __future__ import annotations

import importlib.util
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
