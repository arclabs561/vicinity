from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/download_ann_benchmarks.py"
    )
    spec = importlib.util.spec_from_file_location(
        "download_ann_benchmarks", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_converted_fixture(script: ModuleType, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir()
    hdf5_path = output_dir / "tiny-angular.hdf5"
    hdf5_path.write_bytes(b"cached-hdf5")
    script.write_vec1(output_dir / "train.bin", np.zeros((3, 2), dtype=np.float32))
    script.write_vec1(output_dir / "test.bin", np.ones((2, 2), dtype=np.float32))
    script.write_nbr1(
        output_dir / "neighbors.bin",
        np.array([[0, 1], [1, 2]], dtype=np.int32),
    )
    return {
        "url": "https://example.invalid/tiny-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "recompute_ground_truth": True,
        "expected_bytes": hdf5_path.stat().st_size,
    }


def test_manifestless_outputs_require_explicit_adoption(tmp_path: Path) -> None:
    script = load_script()
    output_dir = tmp_path / "tiny-angular"
    info = write_converted_fixture(script, output_dir)

    with pytest.raises(SystemExit, match="--adopt-existing"):
        script.convert_dataset(
            "tiny-angular",
            info,
            output_dir,
            force=False,
            redownload=False,
        )

    assert not (output_dir / "dataset.json").exists()


def test_adopt_existing_outputs_writes_manifest(tmp_path: Path) -> None:
    script = load_script()
    output_dir = tmp_path / "tiny-angular"
    info = write_converted_fixture(script, output_dir)

    script.convert_dataset(
        "tiny-angular",
        info,
        output_dir,
        force=False,
        redownload=False,
        adopt_existing=True,
    )

    manifest = json.loads((output_dir / "dataset.json").read_text())
    assert manifest["complete"] is True
    assert manifest["settings"] == script.conversion_settings("tiny-angular", info)
    assert manifest["outputs"] == {
        "train_shape": [3, 2],
        "test_shape": [2, 2],
        "neighbors_shape": [2, 2],
    }


def test_adopt_existing_outputs_checks_hdf5_size(tmp_path: Path) -> None:
    script = load_script()
    output_dir = tmp_path / "tiny-angular"
    info = write_converted_fixture(script, output_dir)
    info["expected_bytes"] = 1

    with pytest.raises(SystemExit, match="expected 1"):
        script.convert_dataset(
            "tiny-angular",
            info,
            output_dir,
            force=False,
            redownload=False,
            adopt_existing=True,
        )
