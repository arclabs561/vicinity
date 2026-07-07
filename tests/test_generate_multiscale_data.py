from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/generate_multiscale_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_multiscale_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tiny_scale(script: ModuleType, scale_dir: Path, scale: str) -> dict:
    scale_dir.mkdir()
    script.SCALES[scale] = {
        "n_train": 8,
        "n_test": 3,
        "dim": 4,
        "desc": "Tiny test scale",
    }
    settings = script.scale_settings(scale)
    k = settings["ground_truth_k"]
    script.save_vectors(scale_dir / "train.bin", np.zeros((8, 4), dtype=np.float32))
    script.save_vectors(scale_dir / "test.bin", np.zeros((3, 4), dtype=np.float32))
    script.save_neighbors(scale_dir / "neighbors.bin", np.zeros((3, k), dtype=np.int32))
    script.save_labels(scale_dir / "train_topics.bin", np.zeros(8, dtype=np.int32))
    script.save_f32_array(scale_dir / "test_lids.bin", np.zeros(3, dtype=np.float32))
    script.save_labels(scale_dir / "test_difficulty.bin", np.zeros(3, dtype=np.int32))
    script.save_vectors(
        scale_dir / "test_drift.bin",
        np.zeros((3, 4), dtype=np.float32),
    )
    script.save_neighbors(
        scale_dir / "neighbors_drift.bin",
        np.zeros((3, k), dtype=np.int32),
    )
    script.save_vectors(
        scale_dir / "test_filter.bin",
        np.zeros((3, 4), dtype=np.float32),
    )
    script.save_labels(
        scale_dir / "test_filter_topics.bin",
        np.zeros(3, dtype=np.int32),
    )
    script.save_neighbors(
        scale_dir / "neighbors_filter.bin",
        np.zeros((3, k), dtype=np.int32),
    )
    script.write_json_atomic(scale_dir / "metrics.json", {"meta": {"scale": scale}})
    script.write_json_atomic(
        scale_dir / "multiscale_dataset.json",
        script.output_manifest(scale_dir, settings),
    )
    return settings


def test_matching_scale_outputs_accepts_complete_manifest(tmp_path: Path) -> None:
    script = load_script()
    scale_dir = tmp_path / "Z"
    settings = write_tiny_scale(script, scale_dir, "Z")

    assert script.matching_scale_outputs(scale_dir, settings)


def test_matching_scale_outputs_rejects_truncated_payload(tmp_path: Path) -> None:
    script = load_script()
    scale_dir = tmp_path / "Z"
    settings = write_tiny_scale(script, scale_dir, "Z")
    (scale_dir / "train.bin").write_bytes((scale_dir / "train.bin").read_bytes()[:12])

    assert not script.matching_scale_outputs(scale_dir, settings)
