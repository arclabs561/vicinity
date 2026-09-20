"""Contract tests for the Python wrapper benchmark artifact."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyvicinity")


def test_python_wrapper_benchmark_emits_comparable_phases(tmp_path: Path) -> None:
    output = tmp_path / "wrapper.jsonl"
    script = Path(__file__).parents[1] / "scripts" / "benchmark_python_wrapper.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output",
            str(output),
            "--train",
            "64",
            "--queries",
            "8",
            "--dim",
            "8",
            "--k",
            "3",
            "--batch-size",
            "4",
            "--warmup",
            "2",
            "--ef-search",
            "16",
        ],
        check=True,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["phase"] for row in rows} == {
        "build",
        "single_query",
        "batch_query",
        "numpy_contiguous_conversion",
    }
    assert all(row["result_schema"] == 1 for row in rows)
    assert all(row["algorithm"] == "hnsw" for row in rows)
    assert all(row["seed"] == 42 and row["dim"] == 8 for row in rows)
    batch = next(row for row in rows if row["phase"] == "batch_query")
    assert batch["batches"] == 2
    single = next(row for row in rows if row["phase"] == "single_query")
    assert single["queries_performed"] == 8
    assert single["seconds_per_query_p95"] >= single["seconds_per_query_p50"]
