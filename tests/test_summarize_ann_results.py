from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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
                '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.9,"qps":10}',
                '{"algorithm":"hnsw","storage_mode":"in_memory","recall_at_10":0.8,"qps":20}',
                '{"algorithm":"ivfpq","storage_mode":"mmap","recall_at_10":0.7,"qps":30}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = script.load_summaries([path])

    hnsw = summaries[("glove-25-angular", "hnsw", "in_memory")]
    assert hnsw.rows == 2
    assert hnsw.best_recall == 0.9
    assert hnsw.best_qps == 20.0
    assert summaries[("glove-25-angular", "ivfpq", "mmap")].rows == 1


def test_markdown_table_is_stable(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "rows.jsonl"
    path.write_text(
        '{"algorithm":"hnsw","recall_at_10":1.0,"qps":42}\n',
        encoding="utf-8",
    )

    table = script.markdown_table(script.load_summaries([path]))

    assert "| rows | hnsw | in_memory | 1 | 1.0000 | 42.0 |" in table
