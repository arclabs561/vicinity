from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import NoReturn

import pytest


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts/generate_sparse_mips_smoke_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_sparse_mips_smoke_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compact_pairs_sums_duplicates() -> None:
    script = load_script()

    indices, values = script.compact_pairs([(5, 1.0), (1, 2.0), (5, 0.25)])

    assert indices == [1, 5]
    assert values == [2.0, 1.25]


def test_matching_manifest_reuses_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script()
    output = tmp_path / "sparse-smoke"
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_sparse_mips_smoke_data.py",
            str(output),
            "--train",
            "24",
            "--test",
            "6",
            "--vocab",
            "128",
            "--topics",
            "4",
            "--nnz",
            "12",
            "--k",
            "5",
        ],
    )
    script.main()

    def fail_generate(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("matching outputs should be reused")

    monkeypatch.setattr(script, "generate_dataset", fail_generate)

    script.main()


def test_matching_manifest_rejects_truncated_sparse_payload(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "sparse-smoke"
    output.mkdir()

    class Args:
        train = 24
        test = 6
        vocab = 128
        topics = 4
        nnz = 12
        k = 5
        seed = 42

    settings = script.expected_settings(Args)
    train, test, neighbors = script.generate_dataset(Args)
    train_shape = script.write_spv1(output / "train.spv1", train)
    test_shape = script.write_spv1(output / "test.spv1", test)
    neighbors_shape = script.write_nbr1(output / "neighbors.bin", neighbors)
    script.write_manifest_atomic(
        output / "sparse_mips_dataset.json",
        script.manifest_for(
            settings,
            train_shape=train_shape,
            test_shape=test_shape,
            neighbors_shape=neighbors_shape,
        ),
    )

    train_path = output / "train.spv1"
    train_path.write_bytes(train_path.read_bytes()[:16])

    assert not script.matching_outputs(output, settings)
