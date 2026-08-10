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


def test_repeat_aggregation_reports_median_and_full_spread(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "repeats.jsonl"
    rows = [
        {"_meta": {"dataset": "data/glove", "result_schema": 3}},
        {"algorithm": "hnsw", "storage_mode": "in_memory", "params": {"ef": 50}, "run_id": "r0", "repeat": 0, "recall_at_1": 0.8, "recall_at_10": 0.9, "recall_at_100": 0.95, "qps": 100.0, "latency_us": 10.0},
        {"algorithm": "hnsw", "storage_mode": "in_memory", "params": {"ef": 50}, "run_id": "r1", "repeat": 1, "recall_at_1": 1.0, "recall_at_10": 1.0, "recall_at_100": 0.99, "qps": 80.0, "latency_us": 12.0},
        {"algorithm": "hnsw", "storage_mode": "in_memory", "params": {"ef": 50}, "run_id": "r2", "repeat": 2, "recall_at_1": 0.9, "recall_at_10": 0.95, "recall_at_100": 0.97, "qps": 90.0, "latency_us": 11.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = script.load_summaries([path])[("glove", "hnsw", "in_memory")]
    aggregate = summary.aggregate()
    assert aggregate is not None
    assert aggregate["repeats"] == 3
    assert aggregate["run_ids"] == ["r0", "r1", "r2"]
    assert aggregate["qps_median"] == 90.0
    assert aggregate["qps_spread"] == 20.0
    assert aggregate["recall_at_10_median"] == 0.95
    assert aggregate["recall_at_10_spread"] == pytest.approx(0.1)


def test_repeat_aggregation_dedupes_ids_and_ignores_fast_unqualified_group(
    tmp_path: Path,
) -> None:
    script = load_script()
    path = tmp_path / "selection.jsonl"
    rows = [{"_meta": {"dataset": "data/glove", "result_schema": 3}}]
    rows.extend(
        {
            "algorithm": "hnsw",
            "storage_mode": "in_memory",
            "params": {"ef": 50},
            "run_id": f"qualified-{repeat}",
            "recall_at_10": 0.96,
            "qps": qps,
        }
        for repeat, qps in enumerate((80.0, 90.0, 100.0))
    )
    rows.extend(
        [
            {"algorithm": "hnsw", "storage_mode": "in_memory", "params": {"ef": 5}, "run_id": "fast", "recall_at_10": 0.5, "qps": 1000.0},
            {"algorithm": "hnsw", "storage_mode": "in_memory", "params": {"ef": 5}, "run_id": "fast", "recall_at_10": 0.5, "qps": 2000.0},
            {"algorithm": "hnsw", "storage_mode": "in_memory", "params": {"ef": 50}, "recall_at_10": 1.0, "qps": 9999.0},
        ]
    )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    aggregate = script.load_summaries([path])[("glove", "hnsw", "in_memory")].aggregate()
    assert aggregate is not None
    assert aggregate["params"] == {"ef": 50}
    assert aggregate["repeats"] == 3
    assert aggregate["qps_median"] == 90.0


def test_repeat_aggregation_keeps_cache_states_separate(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "cache-states.jsonl"
    rows = [{"_meta": {"dataset": "data/glove", "result_schema": 3}}]
    for cache_state, qps_values in (
        ("warm", (100.0, 110.0, 120.0)),
        ("reopened", (10.0, 11.0, 12.0)),
    ):
        rows.extend(
            {
                "algorithm": "hnsw",
                "storage_mode": "snapshot_loaded",
                "cache_state": cache_state,
                "params": {"ef": 50},
                "search_k": 100,
                "run_id": f"{cache_state}-{repeat}",
                "recall_at_10": 0.96,
                "qps": qps,
            }
            for repeat, qps in enumerate(qps_values)
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    aggregate = script.load_summaries([path])[
        ("glove", "hnsw", "snapshot_loaded")
    ].aggregate()
    assert aggregate is not None
    assert aggregate["qps_median"] == 110.0
    assert aggregate["run_ids"] == ["warm-0", "warm-1", "warm-2"]


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
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2}}\n'
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


def test_current_schema_only_ignores_non_ann_workload_meta(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "acorn.jsonl"
    path.write_text(
        '{"_meta":{"workload":"acorn_selectivity","result_schema":1,"index_bytes_required":true}}\n'
        '{"algorithm":"acorn","storage_mode":"in_memory","recall_at_10":0.9,"qps":42,"index_bytes":4096}\n',
        encoding="utf-8",
    )

    summaries = script.load_summaries([path], current_schema_only=True)

    assert summaries == {}


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
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"query_limit":1000}}\n'
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


def test_cli_recall_floor_gaps_can_suppress_dominated_scopes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"train_limit":50000,"query_limit":1000}}\n'
        '{"algorithm":"rp_forest","storage_mode":"in_memory","recall_at_10":0.24,"qps":42}\n'
        '{"algorithm":"ivf_avq","storage_mode":"in_memory","recall_at_10":0.20,"qps":24}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"train_limit":50000,"query_limit":500}}\n'
        '{"algorithm":"rp_forest","storage_mode":"in_memory","recall_at_10":0.98,"qps":12}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"train_limit":5000,"query_limit":500}}\n'
        '{"algorithm":"ivf_avq","storage_mode":"in_memory","recall_at_10":0.97,"qps":6}\n',
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
            "--suppress-dominated-recall-gaps",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert [(row["dataset"], row["algorithm"]) for row in output] == [
        ("glove-25-angular[train=50000,queries=1000]", "ivf_avq")
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

    assert {len(line_columns) for line_columns in columns} == {29}
    assert (
        "| rows | - | - | - | - | - | - | - | - | - | - | - | - | hnsw | "
        "in_memory | measured | 1 | 1.0000 | - | 4096 | - | 42.0 | - | "
        "4096 | - | 42.0 | - | 4096 | - |" in table
    )


def test_json_output_preserves_recall_at_10_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"hnsw","recall_at_10":1.0,"qps":42,"index_bytes":4096,'
        '"index_bytes_kind":"heap_estimate"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["summarize_ann_results.py", str(path), "--json"])

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["best_recall_at_10"] == 1.0
    assert output[0]["best_recall_params"] is None
    assert output[0]["best_recall_index_bytes"] == 4096
    assert output[0]["best_recall_index_bytes_kind"] == "heap_estimate"
    assert output[0]["best_index_bytes"] == 4096
    assert output[0]["best_index_bytes_kind"] == "heap_estimate"
    assert output[0]["qps_at_recall_floor"] == 42.0
    assert output[0]["index_bytes_at_recall_floor"] == 4096
    assert output[0]["index_bytes_kind_at_recall_floor"] == "heap_estimate"


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
        json.dumps(
            {
                "dataset": "data/ann-benchmarks/glove-25-angular",
                "metric": "cosine",
                "shape": {"train": 100, "dim": 25},
                "pair_distance_sample": {"p50": 0.4},
                "query_neighbors": {
                    "nearest_distance": {"p50": 0.1},
                    "top2_gap": {"p50": 0.02},
                    "lid_mle": {"p50": 7.5},
                },
                "sampled_relative_contrast": {"p50": 1.8},
                "hubness": {"gini": 0.3},
                "coarse_partition_imbalance": {"count_gini": 0.4},
                "query_splits": [
                    {"kind": "in_distribution"},
                    {"kind": "filtered"},
                ],
            }
        ),
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
    assert output[0]["profile_metric"] == "cosine"
    assert output[0]["profile_train"] == 100
    assert output[0]["profile_dim"] == 25
    assert output[0]["profile_pair_distance_p50"] == 0.4
    assert output[0]["profile_nearest_distance_p50"] == 0.1
    assert output[0]["profile_top2_gap_p50"] == 0.02
    assert output[0]["profile_lid_p50"] == 7.5
    assert output[0]["profile_contrast_p50"] == 1.8
    assert output[0]["profile_hub_gini"] == 0.3
    assert output[0]["profile_coarse_gini"] == 0.4
    assert output[0]["profile_split_kinds"] == "in_distribution,filtered"


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
        '{"algorithm":"hnsw","params":{"ef_search":50},"recall_at_10":0.96,"qps":100,'
        '"index_bytes":1000,"index_bytes_kind":"heap_estimate"}\n'
        '{"algorithm":"hnsw","params":{"ef_search":10},"recall_at_10":0.80,"qps":1000,'
        '"index_bytes":2000,"index_bytes_kind":"snapshot_bytes"}\n',
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
    assert output[0]["best_recall_index_bytes_kind"] == "heap_estimate"
    assert output[0]["best_qps"] == 1000.0
    assert output[0]["best_params"] == {"ef_search": 10}
    assert output[0]["best_index_bytes"] == 2000
    assert output[0]["best_index_bytes_kind"] == "snapshot_bytes"
    assert output[0]["qps_at_recall_floor"] == 100.0
    assert output[0]["params_at_recall_floor"] == {"ef_search": 50}
    assert output[0]["index_bytes_at_recall_floor"] == 1000
    assert output[0]["index_bytes_kind_at_recall_floor"] == "heap_estimate"


def test_json_output_preserves_best_row_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"ivfpq_rerank","storage_mode":"file","recall_at_10":0.90,"qps":10,'
        '"avg_probed_lists":4,"avg_scanned_vectors":128,"avg_code_reads":4,'
        '"avg_code_bytes":512,"avg_partition_reads":4,"avg_partition_bytes":1024,'
        '"avg_vector_reads":80,"avg_vector_bytes":8000,'
        '"avg_page_reads":40,"avg_page_bytes":163840,'
        '"avg_retained_candidates":80}\n'
        '{"algorithm":"ivfpq_rerank","storage_mode":"file","recall_at_10":0.80,"qps":20,'
        '"avg_probed_lists":2,"avg_scanned_vectors":64,"avg_code_reads":2,'
        '"avg_code_bytes":256,"avg_partition_reads":2,"avg_partition_bytes":512,'
        '"avg_vector_reads":20,"avg_vector_bytes":2000,'
        '"avg_page_reads":10,"avg_page_bytes":40960,'
        '"avg_retained_candidates":20}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["summarize_ann_results.py", str(path), "--json"])

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert output[0]["best_qps"] == 20.0
    assert output[0]["best_row_diagnostics"] == {
        "avg_code_bytes": 256.0,
        "avg_code_reads": 2.0,
        "avg_partition_bytes": 512.0,
        "avg_partition_reads": 2.0,
        "avg_page_bytes": 40960.0,
        "avg_page_reads": 10.0,
        "avg_probed_lists": 2.0,
        "avg_retained_candidates": 20.0,
        "avg_scanned_vectors": 64.0,
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


def test_cli_can_require_only_declared_index_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000}}\n'
        '{"algorithm":"legacy_hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000,"index_bytes_required":true}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":24,"index_bytes":4096}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--require-declared-index-bytes",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    by_algorithm = {row["algorithm"]: row for row in output}
    assert by_algorithm["legacy_hnsw"]["best_index_bytes"] is None
    assert not by_algorithm["legacy_hnsw"]["index_bytes_required"]
    assert by_algorithm["hnsw"]["best_index_bytes"] == 4096
    assert by_algorithm["hnsw"]["index_bytes_required"]


def test_cli_can_filter_to_footprint_contract_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"query_limit":1000}}\n'
        '{"algorithm":"legacy_hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n'
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"query_limit":1000,"index_bytes_required":true}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":24,"index_bytes":4096}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--current-schema-only",
            "--footprint-contract-only",
            "--json",
        ],
    )

    script.main()

    output = json.loads(capsys.readouterr().out)
    assert [row["algorithm"] for row in output] == ["hnsw"]
    assert output[0]["best_index_bytes"] == 4096
    assert output[0]["index_bytes_required"]


def test_declared_index_byte_requirement_fails_when_marked_row_lacks_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000,"index_bytes_required":true}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--require-declared-index-bytes",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        script.main()

    assert exc_info.value.code == 1
    assert "glove-25-angular[queries=1000]:hnsw:in_memory" in capsys.readouterr().err


def test_declared_index_byte_requirement_checks_every_marked_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":1000,"index_bytes_required":true}}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":1.0,"qps":100,"index_bytes":4096}\n'
        '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.9,"qps":10}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_ann_results.py",
            str(path),
            "--require-declared-index-bytes",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        script.main()

    assert exc_info.value.code == 1
    assert "glove-25-angular[queries=1000]:hnsw:in_memory" in capsys.readouterr().err


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
    assert ("lsm_churn", "snapshot_loaded") in missing
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
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","result_schema":2,"query_limit":1000}}\n'
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


def test_observed_lsm_churn_requires_snapshot_storage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"_meta":{"dataset":"data/ann-benchmarks/glove-25-angular","query_limit":20}}\n'
        '{"algorithm":"lsm_churn","storage_mode":"in_memory","recall_at_10":1.0,"qps":42}\n',
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
    assert ("lsm_churn", "snapshot_loaded") in missing
    assert ("lsm_churn", "in_memory") not in missing


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
