from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/summarize_selectivity_results.py"
    )
    spec = importlib.util.spec_from_file_location(
        "summarize_selectivity_results", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_results(path: Path) -> None:
    rows = [
        {
            "_meta": {
                "workload": "acorn_selectivity",
                "result_schema": 1,
                "index_bytes_required": True,
                "n": 1200,
                "dim": 32,
                "queries": 100,
                "k": 10,
                "neighbors": 32,
                "ef_search": 200,
                "acorn_max_two_hop_neighbors": 64,
                "fallback_selectivity_threshold": 0.02,
            }
        },
        {
            "algorithm": "acorn",
            "params": {
                "selectivity": 0.1,
                "target_count": 120,
            },
            "recall_at_10": 0.95,
            "storage_mode": "in_memory",
            "cache_state": "warm_after_build",
            "qps": 12345.678,
            "index_bytes": 4096,
            "index_bytes_kind": "synthetic_heap_estimate",
            "p95_us": 90.1,
            "p99_us": 120.2,
            "mean_returned": 9.8,
            "two_hop_invocations": 14,
            "two_hop_nodes_examined": 88,
        },
        {
            "algorithm": "filtered_graph",
            "params": {
                "selectivity": 0.01,
                "target_count": 12,
            },
            "recall_at_10": 0.72,
            "storage_mode": "in_memory",
            "cache_state": "warm_after_build",
            "qps": 4567.0,
            "index_bytes": 2048,
            "index_bytes_kind": "heap_estimate",
            "p95_us": 210.0,
            "p99_us": 300.0,
            "mean_returned": 7.0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_load_rows_keeps_selectivity_curve(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "selectivity.jsonl"
    write_results(path)

    rows = script.load_rows([path])

    assert len(rows) == 2
    assert rows[0].workload == (
        "acorn_selectivity[n=1200,dim=32,queries=100,k=10,neighbors=32,"
        "ef_search=200,acorn_max_two_hop_neighbors=64,"
        "fallback_selectivity_threshold=0.02]"
    )
    assert rows[0].algorithm == "acorn"
    assert rows[0].storage_mode == "in_memory"
    assert rows[0].cache_state == "warm_after_build"
    assert rows[0].selectivity == 0.1
    assert rows[0].target_count == 120
    assert rows[0].recall_key == "recall_at_10"
    assert rows[0].recall == 0.95
    assert rows[0].index_bytes == 4096
    assert rows[0].index_bytes_kind == "synthetic_heap_estimate"
    assert rows[0].index_bytes_required
    assert rows[0].two_hop_invocations == 14.0
    assert rows[1].algorithm == "filtered_graph"
    assert rows[1].selectivity == 0.01


def test_markdown_table_renders_selectivity_columns(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "selectivity.jsonl"
    write_results(path)

    table = script.markdown_table(script.load_rows([path]))

    assert "| Workload | Algorithm | Storage | Cache | Selectivity |" in table
    lines = table.splitlines()
    acorn = next(line for line in lines if "| acorn |" in line)
    filtered = next(line for line in lines if "| filtered_graph |" in line)
    acorn_fragment = (
        "| in_memory | warm_after_build | 0.1 | 120 | 0.95 | 12345.7 | "
        "4096 | synthetic_heap_estimate |"
    )
    assert acorn_fragment in acorn
    assert "| 9.8 | 14 | 88 |" in acorn
    assert "| 0.01 | 12 | 0.72 | 4567.0 | 2048 | heap_estimate |" in filtered
    assert filtered.endswith("| 7 |  |  |")


def test_cli_json_preserves_dynamic_recall_key(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    script = load_script()
    path = tmp_path / "selectivity.jsonl"
    write_results(path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_selectivity_results.py", str(path), "--json"],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["recall_at_10"] == 0.95
    assert output[0]["selectivity"] == 0.1
    assert output[0]["index_bytes"] == 4096
    assert output[0]["index_bytes_kind"] == "synthetic_heap_estimate"
    assert output[0]["index_bytes_required"]
    assert output[1]["algorithm"] == "filtered_graph"


def test_cli_declared_index_byte_requirement_fails_on_missing_row(
    tmp_path: Path, monkeypatch
) -> None:
    script = load_script()
    path = tmp_path / "selectivity.jsonl"
    path.write_text(
        '{"_meta":{"workload":"acorn_selectivity","result_schema":1,"index_bytes_required":true}}\n'
        '{"algorithm":"acorn","params":{"selectivity":0.5,"target_count":10},"storage_mode":"in_memory","recall_at_10":0.9,"qps":42}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_selectivity_results.py",
            str(path),
            "--require-declared-index-bytes",
        ],
    )

    try:
        script.main()
    except SystemExit as exc:
        assert exc.code != 0
        assert "selectivity rows missing index_bytes" in str(exc)
    else:
        raise AssertionError("expected SystemExit")
