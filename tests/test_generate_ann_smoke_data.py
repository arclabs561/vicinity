from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/generate_ann_smoke_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_ann_smoke_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matching_manifest_reuses_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script()
    output = tmp_path / "ann-smoke"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_ann_smoke_data.py",
            str(output),
            "--train",
            "16",
            "--test",
            "5",
            "--dim",
            "4",
            "--k",
            "3",
        ],
    )
    script.main()

    def fail_make_train(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("matching outputs should be reused")

    monkeypatch.setattr(script, "make_train", fail_make_train)

    script.main()


def test_manifest_mismatch_does_not_reuse_outputs(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "ann-smoke"
    output.mkdir()
    train = script.make_train(16, 4)
    test = script.make_test(train, 5)
    neighbors = script.ground_truth(train, test, 3)
    script.write_vec1(output / "train.bin", train)
    script.write_vec1(output / "test.bin", test)
    script.write_nbr1(output / "neighbors.bin", neighbors)
    script.write_manifest_atomic(
        output / "ann_smoke_dataset.json",
        script.manifest_with_outputs(output, script.expected_manifest(16, 5, 4, 3)),
    )

    assert not script.matching_outputs(output, script.expected_manifest(16, 5, 4, 4))


def test_matching_manifest_rejects_truncated_payload(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "ann-smoke"
    output.mkdir()
    train = script.make_train(16, 4)
    test = script.make_test(train, 5)
    neighbors = script.ground_truth(train, test, 3)
    script.write_vec1(output / "train.bin", train)
    script.write_vec1(output / "test.bin", test)
    script.write_nbr1(output / "neighbors.bin", neighbors)
    script.write_manifest_atomic(
        output / "ann_smoke_dataset.json",
        script.manifest_with_outputs(output, script.expected_manifest(16, 5, 4, 3)),
    )

    train_path = output / "train.bin"
    train_path.write_bytes(train_path.read_bytes()[:12])

    assert not script.matching_outputs(output, script.expected_manifest(16, 5, 4, 3))


def test_manifest_records_payload_hashes(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "ann-smoke"
    output.mkdir()
    train = script.make_train(16, 4)
    test = script.make_test(train, 5)
    neighbors = script.ground_truth(train, test, 3)
    script.write_vec1(output / "train.bin", train)
    script.write_vec1(output / "test.bin", test)
    script.write_nbr1(output / "neighbors.bin", neighbors)

    manifest = script.manifest_with_outputs(
        output,
        script.expected_manifest(16, 5, 4, 3),
    )

    assert set(manifest["outputs"]) == {"train.bin", "test.bin", "neighbors.bin"}
    assert (
        manifest["outputs"]["train.bin"]["bytes"]
        == (output / "train.bin").stat().st_size
    )
    assert len(manifest["outputs"]["train.bin"]["sha256"]) == 64


def test_matching_manifest_rejects_same_size_payload_drift(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "ann-smoke"
    output.mkdir()
    train = script.make_train(16, 4)
    test = script.make_test(train, 5)
    neighbors = script.ground_truth(train, test, 3)
    script.write_vec1(output / "train.bin", train)
    script.write_vec1(output / "test.bin", test)
    script.write_nbr1(output / "neighbors.bin", neighbors)
    script.write_manifest_atomic(
        output / "ann_smoke_dataset.json",
        script.manifest_with_outputs(output, script.expected_manifest(16, 5, 4, 3)),
    )

    train_path = output / "train.bin"
    payload = bytearray(train_path.read_bytes())
    payload[-1] ^= 0x01
    train_path.write_bytes(payload)

    assert not script.matching_outputs(output, script.expected_manifest(16, 5, 4, 3))


def test_cli_writes_hashed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script()
    output = tmp_path / "ann-smoke"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_ann_smoke_data.py",
            str(output),
            "--train",
            "16",
            "--test",
            "5",
            "--dim",
            "4",
            "--k",
            "3",
        ],
    )

    script.main()

    manifest = json.loads((output / "ann_smoke_dataset.json").read_text())
    assert "outputs" in manifest
    assert len(manifest["outputs"]["neighbors.bin"]["sha256"]) == 64
