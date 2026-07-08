# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""
Generate cross-algorithm recall-vs-QPS comparison plot from benchmark JSON output.

Usage:
    uv run scripts/plot_comparison.py data/ann-benchmarks/results/glove-25.jsonl
    uv run scripts/plot_comparison.py data/ann-benchmarks/results/*.jsonl

Input: JSONL files where each line is:
    {"algorithm":"hnsw","params":{...},"recall_at_10":0.83,"qps":12345.6,...}

Output: docs/plots/algorithm_comparison_<dataset>.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ALGO_STYLE = {
    "brute": {"color": "#aaaaaa", "marker": "x", "label": "Brute Force"},
    "hnsw": {"color": "#1f77b4", "marker": "o", "label": "HNSW"},
    "hnsw-m16": {"color": "#1f77b4", "marker": "o", "label": "HNSW (M=16)"},
    "hnsw-m32": {"color": "#4e9fd9", "marker": "o", "label": "HNSW (M=32)"},
    "nsw": {"color": "#d62728", "marker": "s", "label": "NSW"},
    "emg": {"color": "#9467bd", "marker": "h", "label": "EMG"},
    "nsg": {"color": "#8c564b", "marker": "p", "label": "NSG"},
    "sng": {"color": "#bcbd22", "marker": "H", "label": "SNG"},
    "vamana": {"color": "#ff7f0e", "marker": "D", "label": "Vamana"},
    "pipnn": {"color": "#17becf", "marker": "P", "label": "PiPNN"},
    "finger": {"color": "#e377c2", "marker": "v", "label": "FINGER"},
    "fresh_graph": {"color": "#7f7f7f", "marker": "<", "label": "FreshGraph"},
    "filtered_graph": {"color": "#393b79", "marker": ">", "label": "FilteredGraph"},
    "ivfpq": {"color": "#2ca02c", "marker": "^", "label": "IVF-PQ"},
    "ivfpq_rerank": {
        "color": "#355f2d",
        "marker": "^",
        "label": "IVF-PQ rerank",
    },
    "ivfpq-1024L": {"color": "#2ca02c", "marker": "^", "label": "IVF-PQ (cb5)"},
    "ivfpq-1024L-cb5": {"color": "#2ca02c", "marker": "^", "label": "IVF-PQ (cb5)"},
    "ivfpq-1024L-cb25": {"color": "#17becf", "marker": "^", "label": "IVF-PQ (cb25)"},
    "ivf_rabitq": {"color": "#637939", "marker": "d", "label": "IVF-RaBitQ"},
    "rp_quant": {"color": "#b5cf6b", "marker": "*", "label": "RpQuant"},
    "scann": {"color": "#9467bd", "marker": "v", "label": "IVF-AVQ"},
    "ivf_avq": {"color": "#9467bd", "marker": "v", "label": "IVF-AVQ"},
    "diskann": {"color": "#333333", "marker": "o", "label": "DiskANN"},
    "kdtree": {"color": "#8c564b", "marker": "+", "label": "KD-Tree"},
    "balltree": {"color": "#e377c2", "marker": "P", "label": "Ball Tree"},
    "rptree": {"color": "#7f7f7f", "marker": "*", "label": "RP-Tree"},
    "rp_forest": {"color": "#bcbd22", "marker": "X", "label": "RP-Forest"},
    "kmeans_tree": {"color": "#17becf", "marker": "h", "label": "K-means Tree"},
}

ALGO_STYLE_ALIASES = {
    "diskann_file": "diskann",
    "diskann_mmap": "diskann",
}


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=9)
    ax.grid(True, linewidth=0.3, color="#dddddd", alpha=0.8)
    ax.set_axisbelow(True)


def pareto_frontier(points):
    """Extract Pareto-optimal points (maximize both recall and QPS).

    A point is Pareto-optimal if no other point has both higher recall
    AND higher QPS. For ANN algorithms, this traces the upper-right
    envelope: as recall increases, QPS typically decreases.
    """
    if not points:
        return []
    # Sort by recall descending; sweep tracking max QPS seen so far
    pts = sorted(points, key=lambda p: -p[0])
    frontier = []
    max_qps = -1
    for recall, qps in pts:
        if qps > max_qps:
            frontier.append((recall, qps))
            max_qps = qps
    # Return sorted by recall ascending for plotting
    return sorted(frontier, key=lambda p: p[0])


def scoped_dataset_name(meta: dict[str, Any]) -> str | None:
    dataset = meta.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        return None

    name = Path(dataset).name
    scope = []
    train_limit = meta.get("train_limit")
    query_limit = meta.get("query_limit")
    if train_limit is not None:
        scope.append(f"train={train_limit}")
    if query_limit is not None:
        scope.append(f"queries={query_limit}")
    if scope:
        name = f"{name}[{','.join(scope)}]"
    return name


def series_key(row: dict[str, Any]) -> str:
    algorithm = row["algorithm"]
    storage_mode = row.get("storage_mode")
    if storage_mode in {"file", "mmap", "snapshot_loaded", "segmented_store"}:
        return f"{algorithm}:{storage_mode}"
    return algorithm


def load_results(paths):
    """Load JSONL results, grouped by dataset and algorithm/storage series."""
    by_dataset = defaultdict(lambda: defaultdict(list))
    for path in [Path(p) for p in paths]:
        current_dataset = path.stem
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = row.get("_meta")
                if isinstance(meta, dict):
                    current_dataset = scoped_dataset_name(meta) or current_dataset
                    continue
                if "algorithm" not in row:
                    continue
                recall = row.get("recall_at_10")
                qps = row.get("qps")
                if not isinstance(recall, (int, float)) or not isinstance(
                    qps, (int, float)
                ):
                    continue
                by_dataset[current_dataset][series_key(row)].append(
                    (float(recall), float(qps))
                )
    return {dataset: dict(by_algo) for dataset, by_algo in by_dataset.items()}


def load_legacy_results(path):
    """Load one JSONL result file, preserving the old helper return shape."""
    by_dataset = load_results([path])
    if len(by_dataset) != 1:
        return {}
    return next(iter(by_dataset.values()))


def plot_one_dataset(dataset, by_algo, output_dir):
    output_dir = Path("docs/plots") if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not by_algo:
        print(f"No results for {dataset}", file=sys.stderr)
        return

    # Wider figure to leave room for legend outside the plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    apply_style(ax)

    for algo, points in sorted(by_algo.items()):
        base_algo = algo.split(":", 1)[0]
        style_algo = ALGO_STYLE_ALIASES.get(base_algo, base_algo)
        style = ALGO_STYLE.get(
            style_algo,
            {"color": "#333333", "marker": ".", "label": base_algo},
        )
        label = style["label"]
        if ":" in algo:
            label = f"{label} ({algo.split(':', 1)[1]})"
        frontier = pareto_frontier(points)

        if len(frontier) == 1:
            # Single point (e.g., brute force)
            ax.scatter(
                [frontier[0][0]],
                [frontier[0][1]],
                color=style["color"],
                marker=style["marker"],
                s=80,
                zorder=5,
                label=label,
            )
        else:
            recalls = [p[0] for p in frontier]
            qps_vals = [p[1] for p in frontier]
            ax.plot(
                recalls,
                qps_vals,
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.5,
                label=label,
                zorder=4,
            )

    ax.set_title(
        "Recall and queries per second tradeoff, up and to the right is better",
        fontsize=10,
        pad=8,
    )
    ax.set_xlabel("Recall@10", fontsize=10)
    ax.set_ylabel("Queries per second (1/s)", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.02)

    all_qps = [q for pts in by_algo.values() for _, q in pts]
    if all_qps:
        ax.set_ylim(min(all_qps) * 0.5, max(all_qps) * 3)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{x:.0f}" if x < 1000 else f"{x / 1000:.0f}K"
        )
    )

    # Legend outside the plot on the right, like ann-benchmarks.com
    ax.legend(
        fontsize=8.5,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
    )

    fig.text(
        0.45,
        -0.02,
        f"Dataset: {dataset}",
        ha="center",
        fontsize=8,
        color="#777777",
    )

    fig.tight_layout()
    out_path = output_dir / f"algorithm_comparison_{dataset}.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_comparison(results_paths, output_dir=None):
    paths = [results_paths] if isinstance(results_paths, (str, Path)) else results_paths
    by_dataset = load_results(paths)
    if not by_dataset:
        print("No results", file=sys.stderr)
        return
    for dataset, by_algo in sorted(by_dataset.items()):
        plot_one_dataset(dataset, by_algo, output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_comparison.py <results.jsonl> [output_dir]", file=sys.stderr)
        sys.exit(1)

    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    output_dir = None
    if len(args) > 1 and not args[-1].endswith(".jsonl"):
        output_dir = args.pop()
    paths = args
    plot_comparison(paths, output_dir)
