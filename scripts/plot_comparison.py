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
    "rptree": {"color": "#7f7f7f", "marker": "*", "label": "RP-Forest"},
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


def load_results(path):
    """Load JSONL results, grouped by algorithm."""
    by_algo = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            algo = d["algorithm"]
            by_algo[algo].append((d["recall_at_10"], d["qps"]))
    return dict(by_algo)


def plot_comparison(results_path, output_dir=None):
    path = Path(results_path)
    if output_dir is None:
        output_dir = Path("docs/plots")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_algo = load_results(path)
    if not by_algo:
        print(f"No results in {path}", file=sys.stderr)
        return

    # Infer dataset name from filename
    dataset = path.stem

    # Wider figure to leave room for legend outside the plot
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    apply_style(ax)

    for algo, points in sorted(by_algo.items()):
        style = ALGO_STYLE.get(algo, {"color": "#333333", "marker": ".", "label": algo})
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
                label=style["label"],
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
                label=style["label"],
                zorder=4,
            )

    ax.set_title(
        f"Recall-Queries per second (1/s) tradeoff — up and to the right is better",
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: plot_comparison.py <results.jsonl> [output_dir]", file=sys.stderr)
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    for path in sys.argv[1:]:
        if path.startswith("-"):
            continue
        plot_comparison(path, output_dir)
