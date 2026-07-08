from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import ModuleType


def load_script(name: str) -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_smoke_dataset(path: Path) -> None:
    smoke = load_script("generate_ann_smoke_data")
    path.mkdir()
    train = smoke.make_train(32, 6)
    test = smoke.make_test(train, 8)
    neighbors = smoke.ground_truth(train, test, 10)
    smoke.write_vec1(path / "train.bin", train)
    smoke.write_vec1(path / "test.bin", test)
    smoke.write_nbr1(path / "neighbors.bin", neighbors)


def write_lbl1(path: Path, labels: list[int]) -> None:
    with path.open("wb") as f:
        f.write(b"LBL1")
        f.write(struct.pack("<I", len(labels)))
        f.write(struct.pack(f"<{len(labels)}I", *labels))


def test_profile_dataset_reports_shape_and_difficulty(tmp_path: Path) -> None:
    script = load_script("profile_ann_dataset")
    dataset = tmp_path / "smoke-angular"
    write_smoke_dataset(dataset)

    profile = script.profile_dataset(
        dataset,
        metric="cosine",
        sample_train=16,
        sample_queries=4,
        pair_samples=20,
        coarse_clusters=4,
        coarse_iters=3,
        lid_k=5,
        seed=1,
    )

    assert profile["shape"] == {
        "dim": 6,
        "ground_truth_k": 10,
        "queries": 8,
        "train": 32,
    }
    assert profile["metric"] == "cosine"
    assert profile["query_splits"] == [
        {
            "dim": 6,
            "ground_truth_k": 10,
            "kind": "in_distribution",
            "name": "base",
            "queries": 8,
        }
    ]
    assert profile["samples"]["train"] == 16
    assert profile["samples"]["queries"] == 4
    assert profile["norms"]["train"]["p50"] > 0.99
    dispersion = profile["coordinate_dispersion_sample"]
    assert dispersion["centroid_norm"] > 0.0
    assert dispersion["dimension_std"]["p50"] >= 0.0
    assert 0.0 < dispersion["variance_effective_dim"] <= 6.0
    assert 0.0 < dispersion["variance_effective_dim_fraction"] <= 1.0
    assert 0.0 <= dispersion["zero_variance_fraction"] <= 1.0
    assert profile["query_neighbors"]["nearest_distance"]["mean"] >= 0.0
    assert profile["query_neighbors"]["top2_gap"]["mean"] >= 0.0
    assert profile["query_neighbors"]["lid_mle"]["mean"] > 0.0
    assert profile["sampled_relative_contrast"]["mean"] > 1.0
    assert profile["hubness"]["topk"] == 10.0
    assert profile["coarse_partition_imbalance"]["clusters"] == 4.0
    assert profile["coarse_partition_imbalance"]["count_max"] > 0.0


def test_cli_writes_json_output(tmp_path: Path, monkeypatch) -> None:
    script = load_script("profile_ann_dataset")
    dataset = tmp_path / "smoke-angular"
    output = tmp_path / "profile.json"
    write_smoke_dataset(dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_ann_dataset.py",
            str(dataset),
            "--sample-train",
            "16",
            "--sample-queries",
            "4",
            "--pair-samples",
            "20",
            "--output",
            str(output),
        ],
    )

    script.main()

    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["dataset"] == str(dataset)
    assert profile["metric"] == "cosine"


def test_profile_dataset_reports_optional_splits(tmp_path: Path) -> None:
    script = load_script("profile_ann_dataset")
    smoke = load_script("generate_ann_smoke_data")
    dataset = tmp_path / "smoke-angular"
    write_smoke_dataset(dataset)
    train = smoke.make_train(32, 6)
    test = smoke.make_test(train, 8)
    neighbors = smoke.ground_truth(train, test, 10)
    smoke.write_vec1(dataset / "test_drift.bin", test)
    smoke.write_nbr1(dataset / "neighbors_drift.bin", neighbors)
    smoke.write_vec1(dataset / "test_filter.bin", test)
    smoke.write_nbr1(dataset / "neighbors_filter.bin", neighbors)
    write_lbl1(dataset / "test_filter_topics.bin", [0, 1, 0, 1, 0, 1, 0, 1])
    write_lbl1(dataset / "test_difficulty.bin", [0, 0, 1, 1, 1, 2, 2, 2])

    profile = script.profile_dataset(
        dataset,
        metric="cosine",
        sample_train=16,
        sample_queries=4,
        pair_samples=20,
        coarse_clusters=4,
        coarse_iters=3,
        lid_k=5,
        seed=1,
    )

    assert profile["query_splits"] == [
        {
            "difficulty_labels": "test_difficulty.bin",
            "dim": 6,
            "ground_truth_k": 10,
            "kind": "in_distribution",
            "name": "base",
            "queries": 8,
        },
        {
            "dim": 6,
            "ground_truth_k": 10,
            "kind": "ood_drift",
            "name": "drift",
            "queries": 8,
        },
        {
            "dim": 6,
            "ground_truth_k": 10,
            "kind": "filtered",
            "label_file": "test_filter_topics.bin",
            "name": "filter",
            "queries": 8,
        },
    ]


def test_infer_metric_uses_manifest(tmp_path: Path) -> None:
    script = load_script("profile_ann_dataset")
    dataset = tmp_path / "custom"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(
        json.dumps({"settings": {"metric": "euclidean"}}),
        encoding="utf-8",
    )

    assert script.infer_metric(dataset, "auto") == "l2"


def test_truncated_vec1_is_rejected(tmp_path: Path) -> None:
    script = load_script("profile_ann_dataset")
    path = tmp_path / "train.bin"
    path.write_bytes(b"VEC1" + (2).to_bytes(4, "little") + (3).to_bytes(4, "little"))

    try:
        script.read_vec1(path)
    except SystemExit as exc:
        assert "expected 36" in str(exc)
    else:
        raise AssertionError("truncated VEC1 should be rejected")
