from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import NoReturn

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
        "expected_sha256": script.sha256_file(hdf5_path),
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


def test_matching_manifest_reuses_converted_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    output_dir = tmp_path / "tiny-angular"
    info = write_converted_fixture(script, output_dir)
    hdf5_path = output_dir / "tiny-angular.hdf5"
    script.write_complete_manifest_from_shapes(
        output_dir,
        "tiny-angular",
        info,
        hdf5_path,
        script.existing_output_shapes(output_dir),
    )

    def fail_download(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("idempotent conversion should not download")

    monkeypatch.setattr(script, "download_file", fail_download)

    script.convert_dataset(
        "tiny-angular",
        info,
        output_dir,
        force=False,
        redownload=False,
    )


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


def test_adopt_existing_outputs_checks_hdf5_sha256(tmp_path: Path) -> None:
    script = load_script()
    output_dir = tmp_path / "tiny-angular"
    info = write_converted_fixture(script, output_dir)
    info["expected_sha256"] = "0" * 64

    with pytest.raises(SystemExit, match="SHA256"):
        script.convert_dataset(
            "tiny-angular",
            info,
            output_dir,
            force=False,
            redownload=False,
            adopt_existing=True,
        )


def test_download_file_checks_cached_hdf5_sha256(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "tiny.hdf5"
    path.write_bytes(b"cached-hdf5")

    with pytest.raises(SystemExit, match="SHA256"):
        script.download_file(
            "https://example.invalid/tiny.hdf5",
            path,
            expected_sha256="0" * 64,
        )


def test_download_dataset_hdf5_downloads_without_converting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = load_script()
    output_dir = tmp_path / "tiny-angular"
    info = {
        "url": "https://example.invalid/tiny-angular.hdf5",
        "metric": "angular",
        "normalize": True,
        "expected_bytes": len(b"hdf5"),
        "expected_sha256": "expected-hash",
    }

    def fake_download(
        _url: str,
        dest: Path,
        *,
        redownload: bool,
        expected_bytes: int | None,
        expected_sha256: str | None,
    ) -> None:
        assert redownload is False
        assert expected_bytes == len(b"hdf5")
        assert expected_sha256 == "expected-hash"
        dest.write_bytes(b"hdf5")

    monkeypatch.setattr(script, "download_file", fake_download)

    script.download_dataset_hdf5(
        "tiny-angular",
        info,
        output_dir,
        redownload=False,
    )

    assert (output_dir / "tiny-angular.hdf5").read_bytes() == b"hdf5"
    assert not (output_dir / "dataset.json").exists()
    assert not (output_dir / "train.bin").exists()


def test_selected_dataset_names_returns_single_dataset() -> None:
    script = load_script()

    assert script.selected_dataset_names("glove-25-angular", False) == [
        "glove-25-angular"
    ]


def test_selected_dataset_names_returns_all_configured_datasets() -> None:
    script = load_script()

    assert script.selected_dataset_names(None, True) == list(script.DATASETS)


def test_selected_dataset_names_rejects_missing_dataset() -> None:
    script = load_script()

    with pytest.raises(SystemExit, match="dataset is required"):
        script.selected_dataset_names(None, False)


def test_selected_dataset_names_rejects_all_with_named_dataset() -> None:
    script = load_script()

    with pytest.raises(SystemExit, match="--all cannot be combined"):
        script.selected_dataset_names("glove-25-angular", True)
