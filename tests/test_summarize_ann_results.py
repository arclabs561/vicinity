from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/summarize_ann_results.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_ann_results", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_summaries_groups_by_dataset_algorithm_and_storage(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular"}}',
                '{"algorithm":"hnsw","storage_mode":"in_memory","params":{"ef_search":50},"recall_at_10":0.97,"qps":10,"index_bytes":1000}',
                '{"algorithm":"hnsw","storage_mode":"in_memory","params":{"ef_search":10},"recall_at_10":0.8,"qps":20,"index_bytes":2000}',
                '{"algorithm":"ivfpq","storage_mode":"mmap","recall_at_10":0.7,"qps":30}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = script.load_summaries([path])

    hnsw = summaries[("glove-25-angular", "hnsw", "in_memory")]
    assert hnsw.rows == 2
    assert hnsw.best_recall == 0.97
    assert hnsw.best_recall_params == {"ef_search": 50}
    assert hnsw.best_recall_index_bytes == 1000
    assert hnsw.best_qps == 20.0
    assert hnsw.best_qps_params == {"ef_search": 10}
    assert hnsw.best_qps_index_bytes == 2000
    assert hnsw.qps_at_recall(0.95) == 10.0
    assert hnsw.params_at_recall(0.95) == {"ef_search": 50}
    assert hnsw.index_bytes_at_recall(0.95) == 1000
    assert hnsw.qps_at_recall(0.99) is None
    assert summaries[("glove-25-angular", "ivfpq", "mmap")].rows == 1


def test_load_summaries_keeps_limit_scopes_separate(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","train_limit":null,"query_limit":null}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.97,"qps":10}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","train_limit":50000,"query_limit":1000}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.80,"qps":20}\n',
        encoding="utf-8",
    )

    summaries = script.load_summaries([path])

    assert summaries[("glove-25-angular", "hnsw", "in_memory")].best_qps == 10.0
    capped = summaries[
        ("glove-25-angular[train=50000,queries=1000]", "hnsw", "in_memory")
    ]
    assert capped.best_qps == 20.0


def test_load_summaries_can_ignore_legacy_rows_without_meta(tmp_path: Path) -> None:
    script = load_script()
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        '{"algorithm":"hnsw","recall_at_10":1.0,"qps":10}\n',
        encoding="utf-8",
    )
    current = tmp_path / "current.jsonl"
    current.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular"}}\n'
        '{"algorithm":"hnsw","storage_mode":"snapshot_loaded","recall_at_10":0.97,"qps":20}\n',
        encoding="utf-8",
    )

    default = script.load_summaries([legacy, current])
    current_only = script.load_summaries(
        [legacy, current],
        current_schema_only=True,
    )

    assert ("legacy", "hnsw", "in_memory") in default
    assert ("legacy", "hnsw", "in_memory") not in current_only
    assert (
        current_only[("glove-25-angular", "hnsw", "snapshot_loaded")].best_qps == 20.0
    )


def test_coverage_rows_marks_expected_missing(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular"}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n',
        encoding="utf-8",
    )

    rows = script.coverage_rows(
        script.load_summaries([path]),
        expected=[("hnsw", "in_memory"), ("store", "segmented_store")],
    )

    by_key = {(row.algorithm, row.storage_mode): row for row in rows}
    assert by_key[("hnsw", "in_memory")].status == "measured"
    assert by_key[("hnsw", "in_memory")].qps_at_recall_floor == 42.0
    assert by_key[("store", "segmented_store")].status == "missing"
    assert by_key[("store", "segmented_store")].best_qps is None


def test_coverage_rows_can_filter_dataset_and_missing_only(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular"}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/sift-128-euclidean"}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":24}\n',
        encoding="utf-8",
    )

    rows = script.coverage_rows(
        script.load_summaries([path]),
        expected=[("hnsw", "in_memory"), ("store", "segmented_store")],
        only_datasets={"glove-25-angular"},
        missing_only=True,
    )

    assert [(row.dataset, row.algorithm, row.status) for row in rows] == [
        ("glove-25-angular", "store", "missing")
    ]


def test_cli_can_emit_recall_floor_gaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.94,"qps":42}\n'
        '{"algorithm":"ivfpq","storage_mode":"in_memory","recall_at_10":0.96,"qps":24}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--expect",
            "store:segmented_store",
            "--recall-gap-only",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert [(row["algorithm"], row["status"]) for row in output] == [
        ("hnsw", "measured")
    ]
    assert output[0]["qps_at_recall_floor"] is None


def test_cli_recall_floor_gaps_ignore_unscoped_current_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.94,"qps":42}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"ivfpq","storage_mode":"in_memory","recall_at_10":0.94,"qps":24}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--current-schema-only",
            "--recall-gap-only",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert [(row["dataset"], row["algorithm"]) for row in output] == [
        ("glove-25-angular[queries=1000]", "ivfpq")
    ]


def test_markdown_table_is_stable(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"hnsw","recall_at_10":1.0,"qps":42,"index_bytes":4096}\n',
        encoding="utf-8",
    )

    table = script.markdown_table(script.coverage_rows(script.load_summaries([path])))
    columns = [line.split("|")[1:-1] for line in table.splitlines()]

    assert {len(line_columns) for line_columns in columns} == {15}
    assert (
        "| rows | - | hnsw | in_memory | measured | 1 | 1.0000 | - | 4096 | "
        "42.0 | - | 4096 | 42.0 | - | 4096 |" in table
    )


def test_json_output_preserves_recall_at_10_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"hnsw","recall_at_10":1.0,"qps":42,"index_bytes":4096}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["summarize_ann_results.py", str(path), "--json"])

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["best_recall_at_10"] == 1.0
    assert output[0]["best_recall_params"] is None
    assert output[0]["best_recall_index_bytes"] == 4096
    assert output[0]["best_index_bytes"] == 4096
    assert output[0]["qps_at_recall_floor"] == 42.0
    assert output[0]["index_bytes_at_recall_floor"] == 4096


def test_json_output_can_include_dataset_profile_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    results = tmp_path / "rows.jsonl"
    results.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular"}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n',
        encoding="utf-8",
    )
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile = profile_dir / "glove-25-angular.json"
    profile.write_text(
        json.dumps({"dataset": "data/ann-benchmarks/glove-25-angular"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(results),
            "--profile-dir",
            str(profile_dir),
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["dataset_profile"] == str(profile)


def test_dataset_profiles_require_exact_dataset_label_for_capped_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    results = tmp_path / "rows.jsonl"
    results.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","train_limit":50000}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n',
        encoding="utf-8",
    )
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "glove-25-angular.json").write_text(
        json.dumps({"dataset": "data/ann-benchmarks/glove-25-angular"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(results),
            "--profile-dir",
            str(profile_dir),
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["dataset"] == "glove-25-angular[train=50000]"
    assert output[0]["dataset_profile"] is None


def test_json_output_uses_recall_floor_for_thresholded_qps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"hnsw","params":{"ef_search":50},"recall_at_10":0.96,"qps":100,"index_bytes":1000}\n'
        '{"algorithm":"hnsw","params":{"ef_search":10},"recall_at_10":0.80,"qps":1000,"index_bytes":2000}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--recall-floor",
            "0.95",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["best_recall_at_10"] == 0.96
    assert output[0]["best_recall_params"] == {"ef_search": 50}
    assert output[0]["best_recall_index_bytes"] == 1000
    assert output[0]["best_qps"] == 1000.0
    assert output[0]["best_params"] == {"ef_search": 10}
    assert output[0]["best_index_bytes"] == 2000
    assert output[0]["qps_at_recall_floor"] == 100.0
    assert output[0]["params_at_recall_floor"] == {"ef_search": 50}
    assert output[0]["index_bytes_at_recall_floor"] == 1000


def test_json_output_preserves_best_row_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"ivfpq_rerank","storage_mode":"file","recall_at_10":0.90,"qps":10,'
        '"avg_vector_reads":80,"avg_vector_bytes":8000,"avg_retained_candidates":80}\n'
        '{"algorithm":"ivfpq_rerank","storage_mode":"file","recall_at_10":0.80,"qps":20,'
        '"avg_vector_reads":20,"avg_vector_bytes":2000,"avg_retained_candidates":20}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["summarize_ann_results.py", str(path), "--json"])

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["best_qps"] == 20.0
    assert output[0]["best_row_diagnostics"] == {
        "avg_retained_candidates": 20.0,
        "avg_vector_bytes": 2000.0,
        "avg_vector_reads": 20.0,
    }


def test_cli_can_require_index_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n'
        '{"algorithm":"ivfpq","storage_mode":"mmap","recall_at_10":1.0,"qps":24,"index_bytes":4096}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--require-index-bytes",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        script.main()

    assert exc_info.value.code == 1
    assert "glove-25-angular[queries=1000]:hnsw:in_memory" in capsys.readouterr().err


def test_cli_index_byte_requirement_ignores_missing_expectations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42,"index_bytes":4096}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--expect",
            "store:segmented_store",
            "--require-index-bytes",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    by_algorithm = {row["algorithm"]: row for row in output}
    assert by_algorithm["hnsw"]["best_index_bytes"] == 4096
    assert by_algorithm["store"]["status"] == "missing"


def test_json_output_preserves_churn_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"inplace_churn","storage_mode":"in_memory","recall_at_10":0.95,"qps":100,'
        '"active_count":64,"update_time_s":0.5,"update_qps":16,"free_slot_ratio":0.125}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["summarize_ann_results.py", str(path), "--json"])

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["best_row_diagnostics"] == {
        "active_count": 64.0,
        "free_slot_ratio": 0.125,
        "update_qps": 16.0,
        "update_time_s": 0.5,
    }


def test_cli_can_emit_standard_storage_missing_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular"}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--expect-standard-storage",
            "--only-dataset",
            "glove-25-angular",
            "--missing-only",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    missing = {(row["algorithm"], row["storage_mode"]) for row in output}
    assert ("hnsw", "snapshot_loaded") in missing
    assert ("diskann_mmap", "mmap") in missing
    assert ("ivfpq_rerank", "file") in missing
    assert ("fresh_graph", "snapshot_loaded") in missing
    assert ("kdtree", "snapshot_loaded") in missing
    assert ("store_snapshot", "segmented_store") in missing
    assert ("store", "segmented_store") in missing
    assert ("sparse_mips", "in_memory") in missing
    assert ("hnsw", "in_memory") not in missing


def test_cli_can_scope_standard_storage_to_observed_algorithms(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"hnsw","recall_at_10":1.0,"qps":42}\n'
        '{"algorithm":"diskann_file","storage_mode":"file","recall_at_10":1.0,"qps":24}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--expect-observed-standard-storage",
            "--missing-only",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    missing = {(row["algorithm"], row["storage_mode"]) for row in output}
    assert ("hnsw", "snapshot_loaded") in missing
    assert ("diskann", "in_memory") in missing
    assert ("diskann_mmap", "mmap") in missing
    assert ("store", "segmented_store") not in missing
    assert ("ivfpq", "file") not in missing


def test_observed_storage_expectations_ignore_unscoped_current_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    historical = tmp_path / "historical.jsonl"
    historical.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.96,"qps":42}\n',
        encoding="utf-8",
    )
    scoped = tmp_path / "scoped.jsonl"
    scoped.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"ivfpq","storage_mode":"in_memory","recall_at_10":0.96,"qps":24}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(historical),
            str(scoped),
            "--current-schema-only",
            "--expect-observed-standard-storage",
            "--missing-only",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    missing = {(row["algorithm"], row["storage_mode"]) for row in output}
    assert ("hnsw", "snapshot_loaded") not in missing
    assert ("ivfpq", "snapshot_loaded") in missing
    assert ("ivfpq", "file") in missing
    assert ("ivfpq", "mmap") in missing


def test_standard_storage_expectations_cover_current_storage_classes() -> None:
    script = load_script()

    rows = set(script.standard_storage_expectations())
    families = script.STANDARD_STORAGE_EXPECTATION_FAMILIES
    family_rows = {
        row for _observed_algorithms, expected_rows in families for row in expected_rows
    }

    for storage_modes, algorithms in script.STORAGE_EXPECTATION_GROUPS.items():
        for algorithm in algorithms:
            for storage_mode in storage_modes:
                assert (algorithm, storage_mode) in rows
    assert ("ivf_avq", "file") in rows
    assert ("ivf_avq", "mmap") in rows
    assert set(script.DISKANN_EXPECTATION_ROWS) <= rows
    assert rows == family_rows
