# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""
Generate theoretical memory scaling plot for vicinity README.

For recall-vs-QPS comparison plots, use scripts/plot_comparison.py instead.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT = Path(__file__).parent

COLORS = {
    "m16": "#1f77b4",
    "m32": "#d62728",
}


def apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=9)
    ax.grid(axis="y", linewidth=0.3, color="#cccccc")
    ax.set_axisbelow(True)



# ===================================================================
# Plot 3: Memory scaling  (theoretical, dim=25 to match GloVe-25)
# ===================================================================
# Formula: raw vectors = N * D * 4 bytes (f32)
#          graph edges ~ N * M * 2 * 4 bytes (M neighbors per node, bidirectional,
#                        u32 ids). Layer overhead adds ~15-20% on top; we use 1.2x.
# We show dim=25 (GloVe) and dim=128 (common embedding dim) side by side.

N = np.array([1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000])
M = 16
GRAPH_FACTOR = 1.2  # accounts for multi-layer overhead


def memory_mb(n, dim, m):
    raw = n * dim * 4  # vector storage
    graph = n * m * 2 * 4 * GRAPH_FACTOR  # neighbor lists
    return raw / 1e6, graph / 1e6


fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
apply_style(ax)

for dim, color, marker, ls in [
    (25, COLORS["m16"], "o", "-"),
    (128, COLORS["m32"], "s", "--"),
]:
    raw, graph = memory_mb(N, dim, M)
    total = raw + graph
    ax.plot(
        N,
        total,
        f"{marker}{ls}",
        color=color,
        markersize=5,
        linewidth=1.5,
        label=f"dim={dim} (total)",
    )
    # Show raw vectors as lighter fill
    ax.fill_between(N, 0, raw, alpha=0.12, color=color)
    ax.plot(N, raw, ls, color=color, linewidth=0.6, alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of vectors", fontsize=10)
ax.set_ylabel("Memory (MB)", fontsize=10)
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda x, _: f"{x / 1e6:.0f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"
    )
)
ax.legend(fontsize=9, frameon=False)

fig.text(
    0.5,
    -0.02,
    "Theoretical memory (M = 16). Shaded region = raw vectors; "
    "gap to line = graph overhead.",
    ha="center",
    fontsize=8,
    color="#555555",
)

fig.tight_layout()
fig.savefig(OUT / "memory_scaling.png", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(f"  wrote {OUT / 'memory_scaling.png'}")
