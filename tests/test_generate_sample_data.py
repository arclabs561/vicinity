from __future__ import annotations

import importlib.util
import struct
from pathlib import Path
from types import ModuleType


def load_script() -> ModuleType:
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/generate_sample_data.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generate_sample_data",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_binary_outputs(script: ModuleType, output: Path) -> None:
    output.mkdir()
    for filename, magic in script.EXPECTED_OUTPUTS.items():
        if magic == b"LBL1":
            (output / filename).write_bytes(magic + struct.pack("<I", 1) + b"\0\0\0\0")
        else:
            (output / filename).write_bytes(
                magic + struct.pack("<II", 1, 1) + b"\0\0\0\0"
            )


def test_manifest_matches_complete_hashed_outputs(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "sample"
    write_binary_outputs(script, output)
    (output / "README.md").write_text("sample data\n")

    script.write_complete_manifest(output, 0.10)

    assert script.all_expected_outputs_exist(output)
    assert script.manifest_matches(output, 0.10)


def test_manifest_rejects_corrupted_output(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "sample"
    write_binary_outputs(script, output)
    readme = output / "README.md"
    readme.write_text("sample data\n")
    script.write_complete_manifest(output, 0.10)

    readme.write_text("changed\n")

    assert not script.manifest_matches(output, 0.10)


def test_readme_is_required_for_complete_outputs(tmp_path: Path) -> None:
    script = load_script()
    output = tmp_path / "sample"
    write_binary_outputs(script, output)

    assert not script.all_expected_outputs_exist(output)
