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

Output: doc/plots/algorithm_comparison_<dataset>.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Style: distill.pub-inspired
ALGO_STYLE = {
    "brute":  {"color": "#999999", "marker": "x",  "label": "Brute Force"},
    "hnsw":   {"color": "#1f77b4", "marker": "o",  "label": "HNSW"},
    "nsw":    {"color": "#d62728", "marker": "s",  "label": "NSW"},
    "ivfpq":  {"color": "#2ca02c", "marker": "^",  "label": "IVF-PQ"},
    "vamana": {"color": "#ff7f0e", "marker": "D",  "label": "Vamana"},
    "scann":  {"color": "#9467bd", "marker": "v",  "label": "ScaNN"},
}


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=9)
    ax.grid(True, linewidth=0.3, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)


def pareto_frontier(points):
    """Extract Pareto-optimal points (maximize both recall and QPS)."""
    # Sort by recall ascending
    pts = sorted(points, key=lambda p: p[0])
    frontier = []
    max_qps = -1
    for recall, qps in pts:
        if qps > max_qps:
            frontier.append((recall, qps))
            max_qps = qps
    return frontier


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
        output_dir = Path("doc/plots")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_algo = load_results(path)
    if not by_algo:
        print(f"No results in {path}", file=sys.stderr)
        return

    # Infer dataset name from filename
    dataset = path.stem

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    apply_style(ax)

    for algo, points in sorted(by_algo.items()):
        style = ALGO_STYLE.get(algo, {"color": "#333333", "marker": ".", "label": algo})
        frontier = pareto_frontier(points)

        if len(frontier) == 1:
            # Single point (e.g., brute force)
            ax.scatter(
                [frontier[0][0]], [frontier[0][1]],
                color=style["color"], marker=style["marker"],
                s=80, zorder=5, label=style["label"],
            )
        else:
            recalls = [p[0] for p in frontier]
            qps_vals = [p[1] for p in frontier]
            ax.plot(
                recalls, qps_vals,
                f'{style["marker"]}-',
                color=style["color"],
                markersize=5, linewidth=1.5,
                label=style["label"],
                zorder=4,
            )
            # Also plot non-frontier points as faded dots
            all_recalls = [p[0] for p in points]
            all_qps = [p[1] for p in points]
            ax.scatter(
                all_recalls, all_qps,
                color=style["color"], alpha=0.15, s=15, zorder=2,
            )

    ax.set_xlabel("Recall@10", fontsize=10)
    ax.set_ylabel("Queries per second (QPS)", fontsize=10)
    ax.set_yscale("log")
    ax.set_xlim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1000 else f"{x/1000:.0f}K"
    ))
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    fig.text(
        0.5, -0.02,
        f"Dataset: {dataset}. Pareto frontier per algorithm. "
        f"Higher and further right is better.",
        ha="center", fontsize=8, color="#555555",
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
