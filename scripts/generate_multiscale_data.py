#!/usr/bin/env python3
"""Generate multi-scale datasets for rigorous ANN benchmarking.

Based on research from:
- ann-benchmarks (erikbern): Standard recall-QPS evaluation
- BigANN 2023 NeurIPS: Filtered, OOD, streaming scenarios at 10M scale
- VIBE: Out-of-distribution evaluation for embeddings

Key insight: Queries should be perturbations of training vectors (like SIFT, GIST, GloVe)
to ensure meaningful nearest neighbor relationships exist.

Scales:
- B (base):  10K vectors - quick validation
- T (test): 100K vectors - meaningful scaling behavior
- P (prod):   1M vectors - production-relevant

Run: uvx --with numpy python scripts/generate_multiscale_data.py [--scale B|T|P|all]
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1
GROUND_TRUTH_K = 100


# =============================================================================
# Core generation functions
# =============================================================================

def generate_clustered_embeddings(
    n: int,
    dim: int,
    n_clusters: int = 100,
    cluster_std: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate embeddings with cluster structure.
    
    This creates a realistic scenario where:
    - Documents cluster by topic/theme
    - Within-cluster similarity is high
    - Between-cluster similarity is lower
    """
    rng = np.random.default_rng(seed)
    
    # Generate cluster centroids spread across the unit sphere
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    
    # Zipf-like cluster sizes (some topics are more common)
    freqs = 1.0 / (np.arange(1, n_clusters + 1, dtype=np.float32) ** 0.8)
    probs = freqs / freqs.sum()
    cluster_ids = rng.choice(n_clusters, n, p=probs, replace=True).astype(np.int32)
    
    # Generate points around their centroids
    vectors = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        noise = rng.standard_normal(dim).astype(np.float32) * cluster_std
        vec = centroids[cluster_ids[i]] + noise
        vectors[i] = vec / np.linalg.norm(vec)
    
    return vectors, cluster_ids, centroids


def generate_queries_as_perturbations(
    train: np.ndarray,
    n_queries: int,
    noise_std: float = 0.12,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate queries by perturbing training vectors.
    
    This is the standard approach used in SIFT, GIST, GloVe benchmarks:
    - Queries have a known "source" vector in the training set
    - Uniform noise level; difficulty is determined POST-HOC by actual LID
    
    Returns: (queries, source_ids)
    
    Note: Difficulty labels are computed separately based on actual LID,
    not assumed from noise level. This ensures LID-stratified evaluation
    is genuine, not a proxy.
    """
    rng = np.random.default_rng(seed)
    n_train, dim = train.shape
    
    # Select source vectors (without replacement for diversity)
    if n_queries > n_train:
        # Allow replacement if more queries than train vectors
        all_sources = rng.choice(n_train, n_queries, replace=True)
    else:
        all_sources = rng.choice(n_train, n_queries, replace=False)
    
    queries = np.empty((n_queries, dim), dtype=np.float32)
    
    for idx in range(n_queries):
        src = all_sources[idx]
        perturbation = rng.standard_normal(dim).astype(np.float32) * noise_std
        vec = train[src] + perturbation
        queries[idx] = vec / np.linalg.norm(vec)
    
    return queries, all_sources.astype(np.int32)


def compute_lid_difficulty_labels(
    lid_values: np.ndarray,
    percentiles: tuple = (33, 66),
) -> np.ndarray:
    """Assign difficulty labels based on actual LID percentiles.
    
    0 = Easy (low LID, dense regions)
    1 = Medium
    2 = Hard (high LID, sparse regions)
    """
    p33, p66 = np.percentile(lid_values, percentiles)
    labels = np.ones(len(lid_values), dtype=np.int32)  # Default medium
    labels[lid_values <= p33] = 0  # Easy
    labels[lid_values >= p66] = 2  # Hard
    return labels


def inject_near_duplicates(
    vectors: np.ndarray,
    labels: np.ndarray,
    frac: float = 0.05,
    noise: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject near-duplicates to model repeated content.
    
    Also updates labels so duplicated vectors have correct labels.
    Returns: (modified_vectors, modified_labels)
    """
    rng = np.random.default_rng(seed)
    n, dim = vectors.shape
    out = vectors.copy()
    out_labels = labels.copy()
    
    n_dups = int(n * frac)
    if n_dups <= 0:
        return out, out_labels
    
    targets = rng.choice(n, n_dups, replace=False)
    sources = rng.choice(n, n_dups, replace=True)
    
    perturbation = rng.standard_normal((n_dups, dim)).astype(np.float32) * noise
    out[targets] = out[sources] + perturbation
    out[targets] /= np.linalg.norm(out[targets], axis=1, keepdims=True)
    
    # IMPORTANT: Update labels for duplicated vectors
    out_labels[targets] = labels[sources]
    
    return out, out_labels


def compute_lid_mle(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 20,
) -> np.ndarray:
    """Compute Local Intrinsic Dimensionality via MLE estimator.
    
    LID_MLE = -k / sum(log(r_i / r_k)) for i in 1..k
    
    Note: Queries are NOT in the training set, so no self-distance to exclude.
    """
    sims = queries @ vectors.T
    dists = np.sqrt(2.0 * (1.0 - np.clip(sims, -1, 1)))
    
    # Get k smallest distances (k nearest neighbors)
    k_dists = np.partition(dists, k-1, axis=1)[:, :k]
    k_dists = np.sort(k_dists, axis=1)
    
    # r_max is the kth distance (farthest of k-NN)
    r_max = k_dists[:, -1:] + 1e-10
    ratios = k_dists / r_max
    ratios = np.clip(ratios, 1e-10, 1.0 - 1e-10)  # Avoid log(0) and log(1)
    
    # MLE estimator: LID = -k / sum(log(r_i / r_k))
    # Note: log(r_k/r_k) = 0, so last term contributes 0
    lid = -(k - 1) / np.sum(np.log(ratios[:, :-1]), axis=1)
    return lid.astype(np.float32)


def simulate_concept_drift(
    queries: np.ndarray,
    train_centroids: np.ndarray,
    query_source_ids: np.ndarray,
    train_cluster_ids: np.ndarray,
    drift_strength: float = 0.2,
    seed: int = 42,
) -> np.ndarray:
    """Simulate concept drift by shifting cluster centroids.
    
    Instead of arbitrary rotation, we shift the 'meaning' of each cluster
    slightly for queries. This models semantic drift (e.g., 'apple' 
    moving from fruit-heavy to tech-heavy).
    
    1. Identify the cluster for each query's source.
    2. Apply a random shift vector to that cluster's centroid.
    3. Move the query in the direction of the new centroid.
    """
    rng = np.random.default_rng(seed)
    n, dim = queries.shape
    
    # Generate random shifts for each cluster
    n_clusters = len(train_centroids)
    cluster_shifts = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    cluster_shifts /= np.linalg.norm(cluster_shifts, axis=1, keepdims=True)
    cluster_shifts *= drift_strength
    
    drifted_queries = queries.copy()
    
    # Get cluster ID for each query's source vector
    query_clusters = train_cluster_ids[query_source_ids]
    
    # Apply shift
    # New Query = Old Query + Cluster Shift
    # This moves the query in the direction of the topic shift
    shifts = cluster_shifts[query_clusters]
    drifted_queries += shifts
    
    # Renormalize
    drifted_queries /= np.linalg.norm(drifted_queries, axis=1, keepdims=True)
    
    return drifted_queries


def compute_ground_truth(train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    """Exact k-NN via brute force."""
    similarities = test @ train.T
    neighbors = np.argsort(-similarities, axis=1)[:, :k]
    return neighbors.astype(np.int32)


def compute_filtered_ground_truth(
    train: np.ndarray,
    test: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact k-NN within label-filtered subsets."""
    label_to_ids = {}
    for lbl in np.unique(train_labels):
        label_to_ids[int(lbl)] = np.where(train_labels == lbl)[0]
    
    neighbors = np.full((len(test), k), -1, dtype=np.int32)
    for i in range(len(test)):
        lbl = int(test_labels[i])
        ids = label_to_ids.get(lbl)
        if ids is None or len(ids) == 0:
            continue
        sims = test[i] @ train[ids].T
        topk_local = np.argsort(-sims)[:k]
        neighbors[i, :len(topk_local)] = ids[topk_local]
    
    return neighbors


def compute_difficulty_metrics(train: np.ndarray, test: np.ndarray) -> dict:
    """Compute dataset difficulty metrics."""
    sims = test @ train.T
    dists = 1 - sims
    d_min = dists.min(axis=1)
    d_mean = dists.mean(axis=1)
    cr = np.where(d_min > 0, d_mean / d_min, np.inf)
    
    lid = compute_lid_mle(train, test, k=20)
    
    # Hubness
    top10 = np.argsort(-sims, axis=1)[:, :10]
    hub_counts = np.bincount(top10.flatten(), minlength=len(train))
    hubness_skew = float(np.mean((hub_counts - hub_counts.mean()) ** 3) / (hub_counts.std() ** 3 + 1e-10))
    
    # Max similarity (sanity check)
    max_sims = sims.max(axis=1)
    
    return {
        "relative_contrast_mean": float(np.mean(cr[np.isfinite(cr)])),
        "relative_contrast_median": float(np.median(cr[np.isfinite(cr)])),
        "lid_mean": float(np.mean(lid)),
        "lid_std": float(np.std(lid)),
        "lid_p25": float(np.percentile(lid, 25)),
        "lid_p75": float(np.percentile(lid, 75)),
        "hubness_skewness": hubness_skew,
        "max_similarity_mean": float(max_sims.mean()),
        "max_similarity_min": float(max_sims.min()),
    }


# =============================================================================
# I/O functions
# =============================================================================

def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_json_atomic(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def save_vectors(path: Path, vectors: np.ndarray) -> None:
    n, d = vectors.shape
    atomic_write(path, b'VEC1' + struct.pack('<II', n, d) + vectors.tobytes())


def save_neighbors(path: Path, neighbors: np.ndarray) -> None:
    n, k = neighbors.shape
    atomic_write(path, b'NBR1' + struct.pack('<II', n, k) + neighbors.tobytes())


def save_labels(path: Path, labels: np.ndarray) -> None:
    labels_u32 = labels.astype(np.uint32)
    atomic_write(path, b'LBL1' + struct.pack('<I', len(labels_u32)) + labels_u32.tobytes())


def save_f32_array(path: Path, arr: np.ndarray) -> None:
    arr_f32 = arr.astype(np.float32)
    atomic_write(path, b'F32A' + struct.pack('<I', len(arr_f32)) + arr_f32.tobytes())


def read_header(path: Path, magic: bytes, fmt: str) -> tuple[int, ...] | None:
    try:
        with path.open("rb") as f:
            header = f.read(len(magic) + struct.calcsize(fmt))
    except FileNotFoundError:
        return None
    if len(header) != len(magic) + struct.calcsize(fmt) or header[:len(magic)] != magic:
        return None
    return struct.unpack(fmt, header[len(magic):])


def scale_settings(scale: str) -> dict:
    cfg = SCALES[scale]
    n_train = cfg["n_train"]
    return {
        "version": MANIFEST_VERSION,
        "scale": scale,
        "config": cfg,
        "n_clusters": min(200, n_train // 50),
        "ground_truth_k": GROUND_TRUTH_K,
        "cluster_std": 0.15,
        "near_duplicate_frac": 0.05,
        "near_duplicate_noise": 0.01,
        "query_noise_std": 0.12,
        "drift_strength": 0.25,
        "seeds": {
            "train": 42,
            "duplicates": 43,
            "queries": 100,
            "drift": 200,
        },
    }


def expected_headers(settings: dict) -> dict[str, tuple[bytes, str, tuple[int, ...]]]:
    cfg = settings["config"]
    n_train = cfg["n_train"]
    n_test = cfg["n_test"]
    dim = cfg["dim"]
    k = settings["ground_truth_k"]
    return {
        "train.bin": (b"VEC1", "<II", (n_train, dim)),
        "test.bin": (b"VEC1", "<II", (n_test, dim)),
        "neighbors.bin": (b"NBR1", "<II", (n_test, k)),
        "train_topics.bin": (b"LBL1", "<I", (n_train,)),
        "test_lids.bin": (b"F32A", "<I", (n_test,)),
        "test_difficulty.bin": (b"LBL1", "<I", (n_test,)),
        "test_drift.bin": (b"VEC1", "<II", (n_test, dim)),
        "neighbors_drift.bin": (b"NBR1", "<II", (n_test, k)),
        "test_filter.bin": (b"VEC1", "<II", (n_test, dim)),
        "test_filter_topics.bin": (b"LBL1", "<I", (n_test,)),
        "neighbors_filter.bin": (b"NBR1", "<II", (n_test, k)),
    }


def expected_file_bytes(settings: dict) -> dict[str, int]:
    cfg = settings["config"]
    n_train = cfg["n_train"]
    n_test = cfg["n_test"]
    dim = cfg["dim"]
    k = settings["ground_truth_k"]
    vec_header = 4 + struct.calcsize("<II")
    nbr_header = 4 + struct.calcsize("<II")
    len_header = 4 + struct.calcsize("<I")
    train_vectors = vec_header + n_train * dim * np.dtype(np.float32).itemsize
    test_vectors = vec_header + n_test * dim * np.dtype(np.float32).itemsize
    neighbors = nbr_header + n_test * k * np.dtype(np.int32).itemsize
    train_labels = len_header + n_train * np.dtype(np.uint32).itemsize
    test_labels = len_header + n_test * np.dtype(np.uint32).itemsize
    test_f32 = len_header + n_test * np.dtype(np.float32).itemsize
    return {
        "train.bin": train_vectors,
        "test.bin": test_vectors,
        "neighbors.bin": neighbors,
        "train_topics.bin": train_labels,
        "test_lids.bin": test_f32,
        "test_difficulty.bin": test_labels,
        "test_drift.bin": test_vectors,
        "neighbors_drift.bin": neighbors,
        "test_filter.bin": test_vectors,
        "test_filter_topics.bin": test_labels,
        "neighbors_filter.bin": neighbors,
    }


def output_manifest(scale_dir: Path, settings: dict) -> dict:
    outputs = {}
    expected_bytes = expected_file_bytes(settings)
    for name, (magic, fmt, expected) in expected_headers(settings).items():
        path = scale_dir / name
        header = read_header(path, magic, fmt)
        if header != expected:
            raise ValueError(f"{name} header mismatch: expected {expected}, got {header}")
        if path.stat().st_size != expected_bytes[name]:
            raise ValueError(
                f"{name} size mismatch: expected {expected_bytes[name]}, got {path.stat().st_size}"
            )
        outputs[name] = {"bytes": path.stat().st_size}
    metrics_path = scale_dir / "metrics.json"
    outputs["metrics.json"] = {"bytes": metrics_path.stat().st_size}
    return {
        "complete": True,
        "settings": settings,
        "outputs": outputs,
    }


def matching_scale_outputs(scale_dir: Path, settings: dict) -> bool:
    manifest_path = scale_dir / "multiscale_dataset.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    if manifest.get("complete") is not True or manifest.get("settings") != settings:
        return False
    outputs = manifest.get("outputs", {})
    expected_bytes = expected_file_bytes(settings)
    for name, (magic, fmt, expected) in expected_headers(settings).items():
        path = scale_dir / name
        if read_header(path, magic, fmt) != expected:
            return False
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes[name]:
            return False
        if outputs.get(name, {}).get("bytes") != actual_bytes:
            return False
    metrics_path = scale_dir / "metrics.json"
    return (
        metrics_path.exists()
        and outputs.get("metrics.json", {}).get("bytes") == metrics_path.stat().st_size
    )


# =============================================================================
# Dataset generation per scale
# =============================================================================

SCALES = {
    "S": {"n_train": 10_000, "n_test": 500, "dim": 384, "desc": "Small (CI/quick)"},
    "M": {"n_train": 100_000, "n_test": 1000, "dim": 384, "desc": "Medium (development)"},
    "L": {"n_train": 1_000_000, "n_test": 2000, "dim": 384, "desc": "Large (production)"},
    "XL": {"n_train": 10_000_000, "n_test": 5000, "dim": 384, "desc": "XLarge (BigANN scale)"},
}


def generate_scale(scale: str, data_dir: Path, force: bool = False) -> dict:
    """Generate all datasets for a given scale."""
    cfg = SCALES[scale]
    n_train = cfg["n_train"]
    n_test = cfg["n_test"]
    dim = cfg["dim"]
    
    scale_dir = data_dir / scale
    scale_dir.mkdir(parents=True, exist_ok=True)
    settings = scale_settings(scale)
    if not force and matching_scale_outputs(scale_dir, settings):
        print(f"\nReusing scale {scale}: {scale_dir}")
        return json.loads((scale_dir / "metrics.json").read_text())
    
    print(f"\n{'='*70}")
    print(f"Scale {scale}: {cfg['desc']}")
    print(f"  Train: {n_train:,} x {dim}, Test: {n_test:,}")
    print(f"{'='*70}")
    
    metrics = {}
    t0 = time.time()
    
    # Adjust parameters based on scale
    n_clusters = min(200, n_train // 50)
    
    # -------------------------------------------------------------------------
    # 1. Generate training data
    # -------------------------------------------------------------------------
    print("\n[1/4] Generating clustered training embeddings...")
    
    train, train_topics, train_centroids = generate_clustered_embeddings(
        n_train, dim,
        n_clusters=n_clusters,
        cluster_std=0.15,
        seed=42,
    )
    
    # Inject near-duplicates (5%) - also updates topic labels
    train, train_topics = inject_near_duplicates(
        train, train_topics, frac=0.05, noise=0.01, seed=43
    )
    
    # -------------------------------------------------------------------------
    # 2. Generate queries as perturbations of training vectors
    # -------------------------------------------------------------------------
    print("  Generating queries (perturbations of training vectors)...")
    
    test, source_ids = generate_queries_as_perturbations(
        train, n_test,
        noise_std=0.12,  # Uniform noise; difficulty determined by actual LID
        seed=100,
    )
    
    # Compute LID for queries
    print("  Computing LID for difficulty stratification...")
    test_lids = compute_lid_mle(train, test, k=20)
    
    # Assign difficulty based on ACTUAL LID, not noise level
    test_difficulty = compute_lid_difficulty_labels(test_lids)
    
    # -------------------------------------------------------------------------
    # 3. Compute ground truth
    # -------------------------------------------------------------------------
    print("  Computing ground truth (k=100)...")
    gt = compute_ground_truth(train, test, k=GROUND_TRUTH_K)
    
    # Verify that source vectors are in top-k for most queries
    source_in_top10 = np.sum([source_ids[i] in gt[i, :10] for i in range(n_test)]) / n_test
    print(f"  Source vector in top-10: {source_in_top10*100:.1f}%")
    
    # Compute metrics
    print("  Computing difficulty metrics...")
    base_metrics = compute_difficulty_metrics(train, test)
    metrics["base"] = base_metrics
    
    # Save base data
    save_vectors(scale_dir / "train.bin", train)
    save_vectors(scale_dir / "test.bin", test)
    save_neighbors(scale_dir / "neighbors.bin", gt)
    save_labels(scale_dir / "train_topics.bin", train_topics)
    save_f32_array(scale_dir / "test_lids.bin", test_lids)
    save_labels(scale_dir / "test_difficulty.bin", test_difficulty)
    
    print(f"  Base: Cr={base_metrics['relative_contrast_mean']:.3f}, " +
          f"LID={base_metrics['lid_mean']:.1f}, " +
          f"MaxSim={base_metrics['max_similarity_mean']:.3f}")
    
    # -------------------------------------------------------------------------
    # 4. OOD/Drift scenario
    # -------------------------------------------------------------------------
    print("\n[2/4] Generating concept drift scenario...")
    
    # We shift the CLUSTERS, not just random directions
    test_drift = simulate_concept_drift(
        test, train_centroids, source_ids, train_topics,
        drift_strength=0.25, seed=200
    )
    gt_drift = compute_ground_truth(train, test_drift, k=GROUND_TRUTH_K)
    drift_metrics = compute_difficulty_metrics(train, test_drift)
    metrics["drift"] = drift_metrics
    
    save_vectors(scale_dir / "test_drift.bin", test_drift)
    save_neighbors(scale_dir / "neighbors_drift.bin", gt_drift)
    
    print(f"  Drift: Cr={drift_metrics['relative_contrast_mean']:.3f}, " +
          f"LID={drift_metrics['lid_mean']:.1f}, " +
          f"MaxSim={drift_metrics['max_similarity_mean']:.3f}")
    
    # -------------------------------------------------------------------------
    # 5. Filtered scenario
    # -------------------------------------------------------------------------
    print("\n[3/4] Generating filtered scenario...")
    
    topic_counts = np.bincount(train_topics, minlength=n_clusters)
    
    # Assign topics to test queries based on their source vectors
    test_filter_topics = train_topics[source_ids].astype(np.int32)
    
    gt_filter = compute_filtered_ground_truth(
        train, test, train_topics, test_filter_topics, k=GROUND_TRUTH_K
    )
    
    save_vectors(scale_dir / "test_filter.bin", test)
    save_labels(scale_dir / "test_filter_topics.bin", test_filter_topics)
    save_neighbors(scale_dir / "neighbors_filter.bin", gt_filter)
    
    valid_filters = np.sum(gt_filter[:, 0] >= 0)
    avg_category_size = np.mean([topic_counts[t] for t in test_filter_topics])
    metrics["filter"] = {
        "valid_queries": int(valid_filters),
        "avg_category_size": float(avg_category_size),
    }
    
    print(f"  Filter: {valid_filters}/{n_test} valid, avg category size={avg_category_size:.0f}")
    
    # -------------------------------------------------------------------------
    # 6. Summary
    # -------------------------------------------------------------------------
    print("\n[4/4] Saving metadata...")
    
    elapsed = time.time() - t0
    total_size = sum(f.stat().st_size for f in scale_dir.glob("*.bin"))
    
    metrics["meta"] = {
        "scale": scale,
        "n_train": n_train,
        "n_test": n_test,
        "dim": dim,
        "n_clusters": n_clusters,
        "generation_time_s": elapsed,
        "total_size_mb": total_size / 1024 / 1024,
    }
    
    write_json_atomic(scale_dir / "metrics.json", metrics)
    write_json_atomic(
        scale_dir / "multiscale_dataset.json",
        output_manifest(scale_dir, settings),
    )
    
    print(f"\n  Total: {total_size / 1024 / 1024:.1f} MB in {elapsed:.1f}s")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate multi-scale ANN benchmark data")
    parser.add_argument(
        "--scale",
        choices=["S", "M", "L", "XL", "all"],
        default="S",
        help="Scale: S=10K, M=100K, L=1M, XL=10M, all=S+M+L"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "multiscale",
        help="Output directory"
    )
    parser.add_argument("--force", action="store_true", help="Regenerate matching outputs")
    args = parser.parse_args()
    
    print("Multi-Scale ANN Benchmark Data Generator")
    print("=" * 70)
    print(f"Output: {args.output}")
    
    scales = ["S", "M", "L"] if args.scale == "all" else [args.scale]
    all_metrics = {}
    
    for scale in scales:
        try:
            metrics = generate_scale(scale, args.output, force=args.force)
            all_metrics[scale] = metrics
        except MemoryError:
            print(f"\nERROR: Out of memory for scale {scale}")
            print("Try running with smaller scale or more RAM")
            sys.exit(1)
    
    # Save combined metrics
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output / "all_metrics.json", all_metrics)
    
    print("\n" + "=" * 70)
    print("Generation complete!")
    print(f"Scales generated: {', '.join(scales)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
