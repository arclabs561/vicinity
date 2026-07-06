#!/usr/bin/env python3
"""Generate bundled sample datasets for benchmarking.

Design based on research into what makes ANN search difficult:

1. Relative Contrast (He et al. 2012): Cr = D_mean / D_min
   - Lower contrast = harder search
   - High dims + dense data = lower contrast

2. Hubness (Radovanovic et al. 2010): Some points become neighbors to many queries
   - Emerges in high dimensions
   - Points near centroid become "hubs"

3. Adversarial queries: Queries between clusters or in sparse regions

Run: uvx --with numpy python scripts/generate_sample_data.py
"""

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1

EXPECTED_OUTPUTS = {
    "quick_train.bin": b"VEC1",
    "quick_test.bin": b"VEC1",
    "quick_neighbors.bin": b"NBR1",
    "bench_train.bin": b"VEC1",
    "bench_test.bin": b"VEC1",
    "bench_neighbors.bin": b"NBR1",
    "hard_train.bin": b"VEC1",
    "hard_test.bin": b"VEC1",
    "hard_neighbors.bin": b"NBR1",
    "hard_train_topics.bin": b"LBL1",
    "hard_test_drift.bin": b"VEC1",
    "hard_neighbors_drift.bin": b"NBR1",
    "hard_test_filter.bin": b"VEC1",
    "hard_neighbors_filter.bin": b"NBR1",
    "hard_test_filter_topics.bin": b"LBL1",
}


def generate_topic_mixture_unit_with_topics(
    n: int,
    dim: int,
    *,
    n_topics: int = 200,
    rank: int = 64,
    topic_scale: float = 1.0,
    topic_noise: float = 0.35,
    global_noise: float = 0.05,
    allowed_topics: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as `generate_topic_mixture_unit`, but also returns topic IDs."""
    rng = np.random.default_rng(seed)
    rank = int(min(rank, dim))

    A = rng.standard_normal((dim, rank)).astype(np.float32)
    Q, _ = np.linalg.qr(A, mode="reduced")

    centroids = rng.standard_normal((n_topics, rank)).astype(np.float32) * topic_scale

    freqs = 1.0 / (np.arange(1, n_topics + 1, dtype=np.float32) ** 1.1)
    probs = freqs / freqs.sum()

    if allowed_topics is None:
        topic_ids = rng.choice(n_topics, size=n, p=probs, replace=True).astype(np.int32)
    else:
        allowed_topics = np.asarray(allowed_topics, dtype=np.int32)
        allowed_probs = probs[allowed_topics]
        allowed_probs = allowed_probs / allowed_probs.sum()
        topic_ids = rng.choice(
            allowed_topics, size=n, p=allowed_probs, replace=True
        ).astype(np.int32)

    latent = (
        centroids[topic_ids]
        + rng.standard_normal((n, rank)).astype(np.float32) * topic_noise
    )
    x = latent @ Q.T
    x += rng.standard_normal((n, dim)).astype(np.float32) * global_noise
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32, copy=False), topic_ids.astype(np.int32, copy=False)


def generate_dense_random_unit(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Dense i.i.d. vectors on the unit sphere.

    Rationale (cosine space):
    - For large `dim`, dot products concentrate tightly near 0 (std ~ 1/sqrt(dim)).
    - Nearest neighbors are only slightly better than random points.
    - That pushes relative contrast Cr toward 1 (hard).

    This deliberately avoids tight clusters, which tend to create very obvious
    nearest neighbors (high Cr, easier recall).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def generate_topic_mixture_unit(
    n: int,
    dim: int,
    *,
    n_topics: int = 200,
    rank: int = 64,
    topic_scale: float = 1.0,
    topic_noise: float = 0.35,
    global_noise: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """A more realistic synthetic embedding distribution.

    Intuition:
    - Real text embeddings tend to be *anisotropic* (variance concentrated in
      a low-rank subspace), and have *topic structure* (mixture components),
      with a *long tail* of topic frequencies.
    - They also contain near-duplicates (reposted docs, boilerplate, templates).

    Construction:
    - Sample an orthonormal basis Q ∈ R^{dim×rank}.
    - Sample topic centroids in the rank-dimensional latent space.
    - Sample topic IDs from a Zipf-like distribution (long tail).
    - Generate points around the chosen topic centroid, project through Q, add
      small global noise, then L2-normalize for cosine search.
    """
    rng = np.random.default_rng(seed)
    rank = int(min(rank, dim))

    # Random orthonormal basis for an anisotropic subspace.
    A = rng.standard_normal((dim, rank)).astype(np.float32)
    Q, _ = np.linalg.qr(A, mode="reduced")  # Q: (dim, rank)

    # Topic centroids in latent space. Use modest scale so topics are confusable.
    centroids = rng.standard_normal((n_topics, rank)).astype(np.float32) * topic_scale

    # Long-tailed topic probabilities (Zipf-ish).
    freqs = 1.0 / (np.arange(1, n_topics + 1, dtype=np.float32) ** 1.1)
    probs = freqs / freqs.sum()
    topic_ids = rng.choice(n_topics, size=n, p=probs, replace=True)

    latent = (
        centroids[topic_ids]
        + rng.standard_normal((n, rank)).astype(np.float32) * topic_noise
    )
    x = latent @ Q.T  # (n, dim)
    x += rng.standard_normal((n, dim)).astype(np.float32) * global_noise
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32, copy=False)


def inject_near_duplicates(
    vectors: np.ndarray,
    *,
    frac: float = 0.10,
    dup_noise: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Overwrite a fraction of vectors with near-duplicates of other vectors.

    This models repeated/templated content in real corpora.
    """
    rng = np.random.default_rng(seed)
    n, dim = vectors.shape
    out = vectors.copy()

    n_dups = int(n * frac)
    if n_dups <= 0:
        return out

    targets = rng.choice(n, n_dups, replace=False)
    sources = rng.choice(n, n_dups, replace=True)

    noise = rng.standard_normal((n_dups, dim)).astype(np.float32) * dup_noise
    out[targets] = out[sources] + noise
    out[targets] /= np.linalg.norm(out[targets], axis=1, keepdims=True)
    return out


def apply_embedding_drift(
    queries: np.ndarray,
    *,
    n_reflections: int = 8,
    mean_shift: float = 0.05,
    noise: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Simulate embedding drift / model mismatch (OOD-ish queries).

    This is intentionally simple (and cheap):
    - add a fixed mean shift direction
    - add small noise
    - renormalize

    Motivation: OOD/DiskANN literature shows ID-optimized graphs degrade sharply
    when queries come from a different distribution (e.g. model upgrade, cross-modal).
    """
    rng = np.random.default_rng(seed)
    n, dim = queries.shape

    shift_dir = rng.standard_normal(dim).astype(np.float32)
    shift_dir /= np.linalg.norm(shift_dir)

    out = queries.astype(np.float32, copy=True)

    # Apply a cheap random orthogonal transform via a small number of
    # Householder reflections. This preserves norm but scrambles directions,
    # modeling a "different embedding space" (OOD) much more strongly than
    # a mild additive shift.
    for _ in range(int(n_reflections)):
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        proj = out @ v  # (n,)
        out = out - 2.0 * proj[:, None] * v[None, :]

    out += shift_dir * mean_shift
    out += rng.standard_normal((n, dim)).astype(np.float32) * noise
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


def save_labels(path: Path, labels: np.ndarray) -> None:
    """Save labels: LBL1 + n (u32) + labels (u32*n)."""
    labels_u32 = labels.astype(np.uint32, copy=False)
    n = labels_u32.shape[0]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(b"LBL1")
        f.write(struct.pack("<I", n))
        f.write(labels_u32.tobytes())
    tmp.replace(path)
    kb = path.stat().st_size / 1024
    print(f"    {path.name}: {n:,} labels = {kb:,.1f} KB")


def compute_ground_truth_filtered_by_label(
    train: np.ndarray,
    test: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Exact k-NN within the subset where train_labels == test_label."""
    train_labels = train_labels.astype(np.int32, copy=False)
    test_labels = test_labels.astype(np.int32, copy=False)

    # Pre-index train ids by label
    label_to_ids: dict[int, np.ndarray] = {}
    for lbl in np.unique(train_labels):
        label_to_ids[int(lbl)] = np.where(train_labels == lbl)[0]

    neighbors = np.empty((len(test), k), dtype=np.int32)
    for i in range(len(test)):
        lbl = int(test_labels[i])
        ids = label_to_ids.get(lbl)
        if ids is None or len(ids) == 0:
            neighbors[i] = -1
            continue
        sims = test[i] @ train[ids].T
        topk_local = np.argsort(-sims)[:k]
        neighbors[i] = ids[topk_local].astype(np.int32)
    return neighbors


def select_low_margin_queries(
    train: np.ndarray,
    candidates: np.ndarray,
    n_select: int,
    *,
    min_top1_sim: float = 0.10,
    seed: int = 42,
) -> np.ndarray:
    """Pick queries with smallest (top1 - top2) similarity margin.

    This makes evaluation harder without relying on ties:
    - if many candidates are nearly as good as the best, approximate search
      struggles to return the exact top-k set.
    """
    rng = np.random.default_rng(seed)
    sims = candidates @ train.T

    # Get top-2 sims per query (unordered) efficiently.
    top2 = np.partition(sims, -2, axis=1)[:, -2:]
    top1 = np.maximum(top2[:, 0], top2[:, 1])
    top2_min = np.minimum(top2[:, 0], top2[:, 1])
    margin = top1 - top2_min

    ok = top1 >= min_top1_sim
    idx = np.where(ok)[0]
    if len(idx) == 0:
        idx = np.arange(len(candidates))

    order = idx[np.argsort(margin[idx])]
    if len(order) >= n_select:
        chosen = order[:n_select]
    else:
        # Pad (rare) with random remaining candidates.
        remaining = n_select - len(order)
        rest = np.setdiff1d(np.arange(len(candidates)), order)
        pad = (
            rng.choice(rest, size=remaining, replace=False)
            if len(rest) >= remaining
            else rng.choice(rest, size=remaining, replace=True)
        )
        chosen = np.concatenate([order, pad])

    return candidates[chosen].astype(np.float32, copy=False)


def generate_low_margin_queries_streaming(
    train: np.ndarray,
    n_select: int,
    *,
    n_candidates: int,
    dim: int,
    min_top1_sim: float | None = 0.10,
    top1_weight: float = 0.10,
    batch_size: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Generate many random unit queries, keep the lowest-margin ones.

    This is the most direct way we know to make the bundled `hard` test set
    harder *without* making the train set bigger:
    - Sample a large candidate pool (e.g. 50k queries).
    - Measure top1-top2 margin against the fixed train set.
    - Keep only the smallest-margin queries.

    Implementation detail:
    - We compute in batches to avoid holding a giant (n_candidates x n_train)
      similarity matrix in memory.
    """
    rng = np.random.default_rng(seed)

    best_queries: list[np.ndarray] = []
    best_scores: list[float] = []

    def push_candidate(q: np.ndarray, score: float):
        # Maintain a fixed-size set of smallest scores.
        if len(best_scores) < n_select:
            best_queries.append(q)
            best_scores.append(score)
            return

        # Replace the current worst (largest) score if this is better.
        worst_idx = int(np.argmax(best_scores))
        if score < best_scores[worst_idx]:
            best_queries[worst_idx] = q
            best_scores[worst_idx] = score

    remaining = n_candidates
    while remaining > 0:
        bs = min(batch_size, remaining)
        remaining -= bs

        # Random unit vectors
        cand = rng.standard_normal((bs, dim)).astype(np.float32)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)

        sims = cand @ train.T
        top2 = np.partition(sims, -2, axis=1)[:, -2:]
        top1 = np.maximum(top2[:, 0], top2[:, 1])
        top2_min = np.minimum(top2[:, 0], top2[:, 1])
        margin = top1 - top2_min

        for i in range(bs):
            if min_top1_sim is not None and top1[i] < min_top1_sim:
                continue
            # Hardness score: prefer (a) tiny margins and (b) low top-1 similarity.
            # This selects queries where NN is both ambiguous and not very "close".
            score = float(margin[i] + top1_weight * top1[i])
            push_candidate(cand[i], score)

    # If we didn't collect enough (min_top1_sim too strict), relax and fill.
    if len(best_queries) < n_select:
        fill = n_select - len(best_queries)
        extra = generate_dense_random_unit(fill, dim, seed=seed + 999)
        best_queries.extend([extra[i] for i in range(fill)])

    return np.stack(best_queries, axis=0).astype(np.float32, copy=False)


def generate_easy_clustered(
    n: int, dim: int, n_clusters: int, cluster_std: float = 0.15, seed: int = 42
) -> np.ndarray:
    """Well-separated clusters. Easy for ANN algorithms."""
    rng = np.random.default_rng(seed)

    # Spread centroids far apart on unit sphere
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    vectors = np.empty((n, dim), dtype=np.float32)
    for i in range(n):
        cluster_idx = i % n_clusters
        noise = rng.standard_normal(dim).astype(np.float32) * cluster_std
        vec = centroids[cluster_idx] + noise
        vectors[i] = vec / np.linalg.norm(vec)

    return vectors


def generate_hard_low_contrast(
    n: int, dim: int, n_clusters: int, seed: int = 42
) -> np.ndarray:
    """Generate data with low relative contrast (hard for ANN).

    Strategy:
    - Many overlapping clusters (high between-cluster variance)
    - Large within-cluster spread (cluster_std = 0.4-0.5)
    - Add "hub" points near global centroid
    - Non-uniform cluster sizes
    """
    rng = np.random.default_rng(seed)

    # Centroids closer together (more overlap)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    # Scale down to bring clusters closer
    centroids *= 0.7

    # Non-uniform cluster sizes (power law-ish)
    cluster_sizes = rng.pareto(a=1.5, size=n_clusters) + 1
    cluster_sizes = (cluster_sizes / cluster_sizes.sum() * n).astype(int)
    cluster_sizes[-1] = n - cluster_sizes[:-1].sum()  # Ensure exact count

    vectors = []

    for cluster_idx, size in enumerate(cluster_sizes):
        # Larger within-cluster spread = more overlap = lower contrast
        cluster_std = 0.35 + rng.uniform(0, 0.15)  # 0.35-0.5

        for _ in range(size):
            noise = rng.standard_normal(dim).astype(np.float32) * cluster_std
            vec = centroids[cluster_idx] + noise
            vec = vec / np.linalg.norm(vec)
            vectors.append(vec)

    vectors = np.array(vectors, dtype=np.float32)

    # Add hub points: 5% of points near global centroid
    # These become "hubs" - appearing as neighbors to many queries
    n_hubs = int(n * 0.05)
    global_centroid = vectors.mean(axis=0)
    global_centroid /= np.linalg.norm(global_centroid)

    hub_indices = rng.choice(n, n_hubs, replace=False)
    for idx in hub_indices:
        noise = rng.standard_normal(dim).astype(np.float32) * 0.1
        vectors[idx] = global_centroid + noise
        vectors[idx] /= np.linalg.norm(vectors[idx])

    return vectors


def generate_adversarial_queries(
    train: np.ndarray, n_queries: int, seed: int = 42
) -> np.ndarray:
    """Generate queries that are hard for ANN algorithms.

    Strategy:
    - 30% between clusters (interpolate between distant points)
    - 30% at cluster boundaries (add large noise to random points)
    - 20% in sparse regions (orthogonal to data manifold)
    - 20% normal (from same distribution as train)
    """
    rng = np.random.default_rng(seed)
    dim = train.shape[1]
    queries = []

    n_between = int(n_queries * 0.3)
    n_boundary = int(n_queries * 0.3)
    n_sparse = int(n_queries * 0.2)
    n_normal = n_queries - n_between - n_boundary - n_sparse

    # Between clusters: interpolate between distant points
    for _ in range(n_between):
        idx1, idx2 = rng.choice(len(train), 2, replace=False)
        # Find somewhat distant pair
        while np.dot(train[idx1], train[idx2]) > 0.5:
            idx1, idx2 = rng.choice(len(train), 2, replace=False)

        alpha = rng.uniform(0.3, 0.7)
        query = alpha * train[idx1] + (1 - alpha) * train[idx2]
        query /= np.linalg.norm(query)
        queries.append(query)

    # Boundary queries: large noise added to existing points
    for _ in range(n_boundary):
        idx = rng.choice(len(train))
        noise = rng.standard_normal(dim).astype(np.float32) * 0.4
        query = train[idx] + noise
        query /= np.linalg.norm(query)
        queries.append(query)

    # Sparse regions: vectors somewhat orthogonal to data
    data_mean = train.mean(axis=0)
    data_mean /= np.linalg.norm(data_mean)

    for _ in range(n_sparse):
        # Random vector with component orthogonal to data mean
        rand_vec = rng.standard_normal(dim).astype(np.float32)
        # Project out component parallel to data mean
        rand_vec -= np.dot(rand_vec, data_mean) * data_mean * 0.7
        rand_vec /= np.linalg.norm(rand_vec)
        queries.append(rand_vec)

    # Normal queries: same distribution as train
    for _ in range(n_normal):
        idx = rng.choice(len(train))
        noise = rng.standard_normal(dim).astype(np.float32) * 0.15
        query = train[idx] + noise
        query /= np.linalg.norm(query)
        queries.append(query)

    return np.array(queries, dtype=np.float32)


def compute_relative_contrast(train: np.ndarray, test: np.ndarray) -> float:
    """Compute average relative contrast: Cr = D_mean / D_min.

    Lower values indicate harder search problems.
    - Easy datasets: Cr > 1.2
    - Medium datasets: 1.05 < Cr < 1.2
    - Hard datasets: Cr < 1.05
    """
    # Use cosine distance = 1 - dot_product for normalized vectors
    similarities = test @ train.T
    distances = 1 - similarities

    d_min = distances.min(axis=1)
    d_mean = distances.mean(axis=1)

    # Avoid division by zero
    cr = np.where(d_min > 0, d_mean / d_min, np.inf)
    return float(cr.mean())


def compute_ground_truth(train: np.ndarray, test: np.ndarray, k: int) -> np.ndarray:
    """Compute exact k-NN via brute force cosine similarity."""
    similarities = test @ train.T
    neighbors = np.argsort(-similarities, axis=1)[:, :k]
    return neighbors.astype(np.int32)


def valid_vec_or_neighbors(path: Path, magic: bytes) -> bool:
    """Return true when a VEC1/NBR1 output has a valid header and byte length."""
    if not path.exists():
        return False
    with path.open("rb") as f:
        header = f.read(12)
    if len(header) != 12 or header[:4] != magic:
        return False
    rows, width = struct.unpack("<II", header[4:])
    return path.stat().st_size == 12 + rows * width * 4


def valid_labels(path: Path) -> bool:
    """Return true when an LBL1 label output has a valid header and byte length."""
    if not path.exists():
        return False
    with path.open("rb") as f:
        header = f.read(8)
    if len(header) != 8 or header[:4] != b"LBL1":
        return False
    (rows,) = struct.unpack("<I", header[4:])
    return path.stat().st_size == 8 + rows * 4


def valid_output(path: Path, magic: bytes) -> bool:
    if magic == b"LBL1":
        return valid_labels(path)
    return valid_vec_or_neighbors(path, magic)


def all_expected_outputs_exist(data_dir: Path) -> bool:
    return all(
        valid_output(data_dir / filename, magic)
        for filename, magic in EXPECTED_OUTPUTS.items()
    )


def manifest_path(data_dir: Path) -> Path:
    return data_dir / "sample_dataset.json"


def generation_settings(hard_dup_frac: float) -> dict:
    return {
        "version": MANIFEST_VERSION,
        "hard_dup_frac": hard_dup_frac,
    }


def manifest_matches(data_dir: Path, hard_dup_frac: float) -> bool:
    path = manifest_path(data_dir)
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return manifest.get("complete") is True and manifest.get(
        "settings"
    ) == generation_settings(hard_dup_frac)


def write_manifest(data_dir: Path, manifest: dict) -> None:
    path = manifest_path(data_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_incomplete_manifest(data_dir: Path, hard_dup_frac: float) -> None:
    write_manifest(
        data_dir,
        {
            "complete": False,
            "settings": generation_settings(hard_dup_frac),
        },
    )


def write_complete_manifest(data_dir: Path, hard_dup_frac: float) -> None:
    outputs = {
        filename: {
            "bytes": (data_dir / filename).stat().st_size,
        }
        for filename in EXPECTED_OUTPUTS
    }
    write_manifest(
        data_dir,
        {
            "complete": True,
            "settings": generation_settings(hard_dup_frac),
            "outputs": outputs,
        },
    )


def default_data_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate repeatable bundled sample datasets for vicinity.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_data_dir(),
        help="Output directory for generated sample files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate files even when the output manifest is complete.",
    )
    parser.add_argument(
        "--hard-dup-frac",
        type=float,
        default=float(os.getenv("VICINITY_HARD_DUP_FRAC", "0.10")),
        help="Near-duplicate fraction for the hard dataset.",
    )
    return parser.parse_args()


def save_vectors(path: Path, vectors: np.ndarray):
    """Save vectors: VEC1 (magic) + n (u32) + dim (u32) + data (f32 * n * dim)."""
    n, d = vectors.shape
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(b"VEC1")
        f.write(struct.pack("<II", n, d))
        f.write(vectors.tobytes())
    tmp.replace(path)
    kb = path.stat().st_size / 1024
    print(f"    {path.name}: {n:,} x {d} = {kb:,.1f} KB")


def save_neighbors(path: Path, neighbors: np.ndarray):
    """Save ground truth: NBR1 (magic) + n (u32) + k (u32) + data (i32 * n * k)."""
    n, k = neighbors.shape
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(b"NBR1")
        f.write(struct.pack("<II", n, k))
        f.write(neighbors.tobytes())
    tmp.replace(path)
    kb = path.stat().st_size / 1024
    print(f"    {path.name}: {n:,} queries x {k} neighbors = {kb:,.1f} KB")


def main():
    args = parse_args()
    data_dir = args.output
    hard_dup_frac = args.hard_dup_frac
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.force and all_expected_outputs_exist(data_dir):
        if manifest_matches(data_dir, hard_dup_frac):
            print(f"Sample data already generated at {data_dir}")
            print("Use --force to regenerate.")
            return
        if not manifest_path(data_dir).exists():
            write_complete_manifest(data_dir, hard_dup_frac)
            print(f"Sample data already generated at {data_dir}")
            print(f"Wrote missing manifest to {manifest_path(data_dir)}")
            print("Use --force to regenerate.")
            return

    write_incomplete_manifest(data_dir, hard_dup_frac)

    print("Generating bundled datasets for vicinity")
    print("=" * 70)

    # =========================================================================
    # Dataset 1: "quick" - For CI and fast iteration (easy)
    # =========================================================================
    print("\n1. quick (2K x 128) - CI, fast iteration")
    print("   Easy: well-separated clusters, standard queries")

    train = generate_easy_clustered(2_000, 128, n_clusters=20, seed=42)
    test = generate_easy_clustered(200, 128, n_clusters=20, seed=100)
    gt = compute_ground_truth(train, test, k=100)
    cr = compute_relative_contrast(train, test)
    cr_quick = cr

    save_vectors(data_dir / "quick_train.bin", train)
    save_vectors(data_dir / "quick_test.bin", test)
    save_neighbors(data_dir / "quick_neighbors.bin", gt)
    print(f"    Relative contrast: {cr:.3f} (>1.1 = easy)")

    # =========================================================================
    # Dataset 2: "bench" - Realistic modern embeddings (medium)
    # =========================================================================
    print("\n2. bench (10K x 384) - Realistic embedding dimensions")
    print("   Medium: moderate overlap, mixed queries")

    train = generate_easy_clustered(
        10_000, 384, n_clusters=50, cluster_std=0.25, seed=42
    )
    test = generate_adversarial_queries(train, 500, seed=200)
    gt = compute_ground_truth(train, test, k=100)
    cr = compute_relative_contrast(train, test)
    cr_bench = cr

    save_vectors(data_dir / "bench_train.bin", train)
    save_vectors(data_dir / "bench_test.bin", test)
    save_neighbors(data_dir / "bench_neighbors.bin", gt)
    print(f"    Relative contrast: {cr:.3f} (1.05-1.1 = medium)")

    # =========================================================================
    # Dataset 3: "hard" - Deliberately difficult (hard)
    # =========================================================================
    # Based on: He et al. "On the Difficulty of Nearest Neighbor Search" (ICML 2012)
    print("\n3. hard (10K x 768) - Stress test for ANN algorithms")
    print("   Hard: topic mixture + anisotropy + duplicates + hard tail queries")

    # Key hardness knob for cosine search:
    # - realistic anisotropy + confusable topics (mixture)
    # - some near-duplicates (common in real corpora)
    # Research-informed parameters (He et al. 2012, GIST-960 characteristics):
    # - Lower topic_scale = topics closer together = more overlap = lower Cr
    # - Higher topic_noise = more within-topic spread = harder disambiguation
    # - rank=48 (lower than 64) = more concentration in fewer dimensions
    train, train_topics = generate_topic_mixture_unit_with_topics(
        10_000,
        768,
        n_topics=200,
        rank=48,  # Lower rank = more anisotropy = harder (was 64)
        topic_scale=0.7,  # Closer topics = more overlap (was 0.9)
        topic_noise=0.50,  # More within-topic spread (was 0.40)
        global_noise=0.08,  # Slightly more ambient noise (was 0.05)
        seed=42,
    )
    train = inject_near_duplicates(train, frac=hard_dup_frac, dup_noise=0.008, seed=43)

    # Topics used for filtering scenarios.
    # Ensure categories have enough items for k=100 ground truth.
    topic_counts = np.bincount(train_topics, minlength=200)
    eligible_topics = np.where(topic_counts >= 200)[0]
    if len(eligible_topics) == 0:
        eligible_topics = np.arange(200)

    # Hardest-tail query selection by top1-top2 margin.
    # We sample many random candidates and keep the most ambiguous ones.
    # Mix: mostly in-distribution queries + a LARGER hard tail (stress test).
    n_test = 500
    n_hard_tail = 200  # More hard-tail queries (was 100)
    n_normal = n_test - n_hard_tail

    # Test queries should match train distribution for fair evaluation
    test_normal, test_normal_topics = generate_topic_mixture_unit_with_topics(
        n_normal,
        768,
        n_topics=200,
        rank=48,  # Match train
        topic_scale=0.7,  # Match train
        topic_noise=0.50,  # Match train
        global_noise=0.08,  # Match train
        allowed_topics=eligible_topics,
        seed=300,
    )

    # Hard tail: many candidates from the same generator; keep ambiguous ones.
    # Use larger candidate pool for better selection (was 50k)
    candidates, candidates_topics = generate_topic_mixture_unit_with_topics(
        100_000,  # Larger pool for better hard-tail selection
        768,
        n_topics=200,
        rank=48,  # Match train
        topic_scale=0.7,  # Match train
        topic_noise=0.50,  # Match train
        global_noise=0.08,  # Match train
        allowed_topics=eligible_topics,
        seed=301,
    )
    # Keep the topics for filter scenario by selecting on the same indices.
    # Research-based hard-tail selection: smallest (top1 - top2) margin queries.
    sims = candidates @ train.T
    top2 = np.partition(sims, -2, axis=1)[:, -2:]
    top1 = np.maximum(top2[:, 0], top2[:, 1])
    top2_min = np.minimum(top2[:, 0], top2[:, 1])
    margin = top1 - top2_min
    # Lower similarity threshold: we want queries that are HARD, not just close
    ok = top1 >= 0.10  # Was 0.15 - allow queries with lower top-1 similarity
    idx = np.where(ok)[0]
    if len(idx) < n_hard_tail:
        idx = np.arange(len(candidates))
    # Sort by margin primarily, but also penalize queries with very high top-1
    # (those are easy even with small margins because they're so close to train)
    composite_score = margin[idx] + 0.05 * top1[idx]
    hard_order = idx[np.argsort(composite_score)]
    chosen = hard_order[:n_hard_tail]
    test_hard = candidates[chosen]
    test_hard_topics = candidates_topics[chosen]

    median_margin = float(np.median(margin[chosen]))
    print(f"    Hard-tail median margin: {median_margin:.4f}")

    test = np.vstack([test_normal, test_hard]).astype(np.float32, copy=False)
    test_topics = np.concatenate([test_normal_topics, test_hard_topics]).astype(
        np.int32, copy=False
    )
    gt = compute_ground_truth(train, test, k=100)
    cr = compute_relative_contrast(train, test)
    cr_hard = cr

    save_vectors(data_dir / "hard_train.bin", train)
    save_vectors(data_dir / "hard_test.bin", test)
    save_neighbors(data_dir / "hard_neighbors.bin", gt)
    save_labels(data_dir / "hard_train_topics.bin", train_topics)
    print(f"    Relative contrast: {cr:.3f} (closer to 1 = harder)")
    print(f"    hard dup frac: {hard_dup_frac:.2f}")

    # Scenario A: OOD-ish queries (model mismatch / cross-modal).
    # Take the BASE test queries and apply embedding drift transformation.
    # Key insight from OOD-DiskANN (Jaiswal et al.): graphs built on in-distribution
    # data degrade sharply when queries come from a different embedding space.
    # The transformation should be STRONG enough to create real distribution shift.
    test_drift = apply_embedding_drift(
        test,
        n_reflections=12,  # Strong transformation (was 8)
        mean_shift=0.15,  # More aggressive shift (was 0.05)
        noise=0.10,  # More noise (was 0.05)
        seed=400,
    )
    gt_drift = compute_ground_truth(train, test_drift, k=100)
    cr_drift = compute_relative_contrast(train, test_drift)
    print(f"    Drift contrast: {cr_drift:.3f} (should be lower than base)")
    save_vectors(data_dir / "hard_test_drift.bin", test_drift)
    save_neighbors(data_dir / "hard_neighbors_drift.bin", gt_drift)

    # Scenario B: filtered queries (topic equality filter)
    # Ground truth computed within each topic.
    gt_filter = compute_ground_truth_filtered_by_label(
        train, test, train_topics, test_topics, k=100
    )
    save_vectors(data_dir / "hard_test_filter.bin", test)
    save_neighbors(data_dir / "hard_neighbors_filter.bin", gt_filter)
    save_labels(data_dir / "hard_test_filter_topics.bin", test_topics)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    total_size = sum(f.stat().st_size for f in data_dir.glob("*.bin"))
    print(f"Total: {total_size / 1024 / 1024:.1f} MB")
    print(f"Location: {data_dir}")

    # Generate README
    readme_content = f"""# Bundled Sample Datasets

Pre-generated datasets for benchmarking without external downloads.

## Datasets

| Name | Train | Test | Dims | Size | Difficulty | Relative Contrast (measured) |
|------|-------|------|------|------|------------|-------------------|
| quick | 2,000 | 200 | 128 | ~1MB | Easy | {cr_quick:.3f} |
| bench | 10,000 | 500 | 384 | ~16MB | Medium | {cr_bench:.3f} |
| hard | 10,000 | 500 | 768 | ~31MB | Hard | {cr_hard:.3f} |

## What Makes "hard" Hard (and realistic)?

This dataset aims to resemble *real embedding corpora* rather than being
purely adversarial.

1. **Anisotropy + topic mixture**
   - Vectors live mostly in a low-rank subspace (rank≈64 inside 768d).
   - Topics follow a Zipf-like long tail (few large topics, many small).

2. **Near-duplicates**
   - We inject near-duplicate vectors to mimic repeated/templated content.
   - Controlled by `VICINITY_HARD_DUP_FRAC` (default: 0.10).

3. **Hard-tail queries**
   - Most queries are in-distribution.
   - A smaller slice is selected for tiny top1–top2 similarity margins.

4. **High dimensionality**
   - 768 dims (matches many transformer embedding models).

## Measuring Recall

These datasets are synthetic and we occasionally retune them. Treat any
“expected recall” numbers as stale unless they come from a fresh run.

```sh
cargo run --example 03_quick_benchmark --release
VICINITY_DATASET=hard cargo run --example 03_quick_benchmark --release

# Scenarios
VICINITY_DATASET=hard VICINITY_TEST_VARIANT=drift \
  cargo run --example 03_quick_benchmark --release
VICINITY_DATASET=hard VICINITY_TEST_VARIANT=filter \
  cargo run --example 03_quick_benchmark --release
```

## Usage

```sh
# Easy (CI)
VICINITY_DATASET=quick cargo run --example 03_quick_benchmark --release

# Medium (default)
cargo run --example 03_quick_benchmark --release

# Hard (stress test)
VICINITY_DATASET=hard cargo run --example 03_quick_benchmark --release
```

## File Format

```
Vectors: VEC1 (4 bytes) + n (u32) + dim (u32) + data (f32 * n * dim)
Neighbors: NBR1 (4 bytes) + n (u32) + k (u32) + data (i32 * n * k)
Labels: LBL1 (4 bytes) + n (u32) + labels (u32 * n)
```

## Regenerating

```sh
uvx --with numpy python scripts/generate_sample_data.py
```

## References

- He, Kumar, Chang. "On the Difficulty of Nearest Neighbor Search" (ICML 2012)
- Radovanovic et al. "Hubs in Space" (JMLR 2010)
- Patel et al. "ACORN" (arXiv 2403.04871)
- Jaiswal et al. "OOD-DiskANN" (arXiv 2211.12850)
- Iff et al. "Benchmarking Filtered ANN on transformer embeddings" (arXiv 2507.21989)
"""
    readme_path = data_dir / "README.md"
    readme_tmp = readme_path.with_suffix(readme_path.suffix + ".tmp")
    readme_tmp.write_text(readme_content)
    readme_tmp.replace(readme_path)
    write_complete_manifest(data_dir, hard_dup_frac)
    print(f"\nGenerated {data_dir / 'README.md'}")


if __name__ == "__main__":
    main()
