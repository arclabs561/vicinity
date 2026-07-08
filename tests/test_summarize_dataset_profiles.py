from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/summarize_dataset_profiles.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_dataset_profiles", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def profile() -> dict:
    return {
        "coarse_partition_imbalance": {"count_gini": 0.25},
        "dataset": "data/ann-benchmarks/glove-25-angular",
        "hubness": {"gini": 0.9},
        "metric": "cosine",
        "pair_distance_sample": {"p50": 0.8},
        "query_neighbors": {
            "lid_mle": {"p50": 8.2},
            "nearest_distance": {"p50": 0.1},
            "top2_gap": {"p50": 0.01},
        },
        "query_splits": [
            {"kind": "in_distribution", "name": "base"},
            {"kind": "ood_drift", "name": "drift"},
        ],
        "sampled_relative_contrast": {"p50": 7.9},
        "shape": {"dim": 25, "train": 1_183_514},
    }


def test_markdown_table_contains_profile_fields() -> None:
    script = load_script()

    table = script.markdown_table([profile()])

    assert "| Dataset | Metric | Train | Dim |" in table
    assert (
        "| glove-25-angular | cosine | 1,183,514 | 25 | 0.8 | 0.1 | 0.01 | "
        "8.2 | 7.9 | 0.9 | 0.25 | in_distribution,ood_drift |"
    ) in table


def test_cli_json_outputs_rows(tmp_path: Path, capsys, monkeypatch) -> None:
    script = load_script()
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_dataset_profiles.py", str(path), "--json"],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0][0] == "glove-25-angular"
    assert output[0][-1] == "in_distribution,ood_drift"


def test_missing_fields_render_as_empty_cells() -> None:
    script = load_script()

    table = script.markdown_table([{"dataset": "minimal"}])

    assert "| minimal |  |  |  |" in table
