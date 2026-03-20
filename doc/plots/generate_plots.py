# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.8"]
# ///
"""
Generate README plots for vicinity HNSW benchmarks.

Data sources:
  - doc/benchmark-results.md (GloVe-25, 1.2M vectors, 25 dims, cosine)
  - Theoretical memory formulas: raw = N*D*4, graph ~ N*M*4*avg_layers
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Style: distill.pub-inspired -- white background, minimal chrome, thin spines
# ---------------------------------------------------------------------------
COLORS = {
    "m16": "#1f77b4",  # muted blue
    "m32": "#d62728",  # muted red
    "raw": "#aec7e8",  # light blue (stacked area)
    "graph": "#1f77b4",  # blue (stacked area)
    "total": "#333333",  # dark grey for total line
}


def apply_style(ax):
    """Minimal distill.pub-like axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=9)
    ax.grid(axis="y", linewidth=0.3, color="#cccccc")
    ax.set_axisbelow(True)


# ===================================================================
# Plot 1: Recall@10 vs ef_search  (GloVe-25, M=16 and M=32)
# ===================================================================
# Real data from doc/benchmark-results.md
ef_vals = np.array([10, 20, 50, 100, 200, 400])
recall_m16 = np.array([58.0, 70.4, 83.1, 89.9, 94.3, 96.8])
recall_m32 = np.array([72.8, 83.1, 92.1, 96.2, 98.3, 99.2])

fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
apply_style(ax)

ax.plot(
    ef_vals,
    recall_m16,
    "o-",
    color=COLORS["m16"],
    markersize=5,
    linewidth=1.5,
    label="M = 16",
)
ax.plot(
    ef_vals,
    recall_m32,
    "s-",
    color=COLORS["m32"],
    markersize=5,
    linewidth=1.5,
    label="M = 32",
)
ax.axhline(95, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
ax.text(405, 95.6, "95%", fontsize=8, color="#999999", va="bottom")

ax.set_xlabel("ef_search", fontsize=10)
ax.set_ylabel("Recall@10 (%)", fontsize=10)
ax.set_xlim(0, 430)
ax.set_ylim(50, 100)
ax.legend(fontsize=9, frameon=False)

fig.text(
    0.5,
    -0.02,
    "GloVe-25 (1.2M vectors, 25-d, cosine). Higher M gives better recall "
    "at each ef_search budget.",
    ha="center",
    fontsize=8,
    color="#555555",
)

fig.tight_layout()
fig.savefig(OUT / "recall_vs_ef.png", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(f"  wrote {OUT / 'recall_vs_ef.png'}")


# ===================================================================
# Plot 2: Build time vs M  (GloVe-25, 1.2M vectors)
# ===================================================================
# Real data from doc/benchmark-results.md
m_vals = [16, 32]
build_secs = [270, 505]  # seconds
vecs_per_sec = [4377, 2343]

fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=150)
apply_style(ax)

bar_x = np.arange(len(m_vals))
bars = ax.bar(
    bar_x, build_secs, width=0.45, color=COLORS["m16"], edgecolor="white", linewidth=0.5
)

# Annotate with throughput
for i, (b, vps) in enumerate(zip(bars, vecs_per_sec)):
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + 12,
        f"{vps:,} vec/s",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#333333",
    )

ax.set_xticks(bar_x)
ax.set_xticklabels([f"M = {m}" for m in m_vals], fontsize=10)
ax.set_ylabel("Build time (seconds)", fontsize=10)
ax.set_ylim(0, 600)

fig.text(
    0.5,
    -0.02,
    "GloVe-25 (1.2M vectors, 25-d). ef_construction = 200, single-threaded.",
    ha="center",
    fontsize=8,
    color="#555555",
)

fig.tight_layout()
fig.savefig(OUT / "build_time_vs_m.png", bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(f"  wrote {OUT / 'build_time_vs_m'}")


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
