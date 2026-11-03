#!/usr/bin/env python3
"""Render the benchmark atlas: every measured family, dataset and tier we have.

Two input schemas are read, because the repository records benchmarks in two
shapes:

* the unified rows written by ``examples/all_families.rs`` and
  ``examples/large_scale.rs`` (``family``/``dataset``/``knob``/``knob_value``),
* the older per-dataset rows under ``docs/*.jsonl`` (``algorithm``/``qps``/
  ``recall_at_10``).

Both are normalised into one ``Row`` before plotting. Every figure carries its
source file, size and protocol in the subtitle, so a plot cannot be quoted
without its context.

Run:

    uv run --with matplotlib --with numpy python scripts/plot_atlas.py

Outputs ``docs/plots/atlas/*.png``, which ``docs/benchmark-atlas.md`` embeds.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "plots" / "atlas"

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    family: str
    dataset: str
    knob: str
    knob_value: float
    recall: float
    qps: float | None
    p50_us: float | None
    p95_us: float | None
    p99_us: float | None
    build_s: float | None
    mem_mib: float | None
    rss_mib: float | None
    dists_per_query: float | None
    n: int | None
    source: str


def _num(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def read_unified(path: Path, meta: dict) -> list[Row]:
    """Rows from `benchmarks/*.jsonl` (family/dataset/knob schema)."""
    rows: list[Row] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if raw.get("_meta"):
            continue
        recall = _num(raw.get("recall_at_10", raw.get("recall")))
        if recall is None:
            continue
        rows.append(
            Row(
                family=str(raw.get("family", "?")),
                dataset=str(raw.get("dataset", meta.get("dataset") or "?")).split("@")[
                    0
                ],
                knob=str(raw.get("knob", "?")),
                knob_value=_num(raw.get("knob_value")) or 0.0,
                recall=recall,
                qps=_num(raw.get("qps")),
                p50_us=_num(raw.get("p50_us")),
                p95_us=_num(raw.get("p95_us")),
                p99_us=_num(raw.get("p99_us")),
                build_s=_num(raw.get("build_s")),
                mem_mib=_num(raw.get("mem_mib")),
                rss_mib=_num(raw.get("rss_mib")),
                dists_per_query=_num(raw.get("dists_per_query")),
                n=raw.get("n") or meta.get("n"),
                source=path.name,
            )
        )
    return rows


def read_algorithms(path: Path) -> tuple[list[Row], dict]:
    """Rows from the older `docs/*.jsonl` (algorithm/qps schema)."""
    meta: dict = {}
    rows: list[Row] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        if raw.get("_meta"):
            meta = raw["_meta"]
            continue
        recall = _num(raw.get("recall_at_10"))
        if recall is None:
            continue
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        ef = _num(raw.get("ef_search"))
        if ef is None:
            ef = _num(params.get("ef_search")) or _num(params.get("nprobe")) or 0.0
        rows.append(
            Row(
                family=str(raw.get("algorithm", "?")),
                dataset=path.stem,
                knob="ef_search",
                knob_value=ef,
                recall=recall,
                qps=_num(raw.get("qps")),
                p50_us=_num(raw.get("p50_us")),
                p95_us=_num(raw.get("p95_us")),
                p99_us=_num(raw.get("p99_us")),
                build_s=_num(raw.get("build_time_s") or raw.get("build_time_secs")),
                mem_mib=None,
                rss_mib=None,
                dists_per_query=None,
                n=raw.get("n_vectors") or meta.get("indexed_vectors"),
                source=path.name,
            )
        )
    return rows, meta


def load_all() -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = {}
    for path in sorted((ROOT / "benchmarks").glob("*.jsonl")):
        meta = {}
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            if raw.get("_meta"):
                meta = raw
                break
        out[path.name] = read_unified(path, meta)
    for name in ("glove-25-angular", "sift-128-euclidean", "gist-960-euclidean"):
        path = ROOT / "docs" / f"{name}.jsonl"
        if path.exists():
            out[path.name], _ = read_algorithms(path)
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pareto(rows: list[Row]) -> list[Row]:
    """Recall-ascending non-dominated set on (recall, qps)."""
    best = float("-inf")
    front: list[Row] = []
    for row in sorted(rows, key=lambda r: (r.recall, -(r.qps or 0.0))):
        qps = row.qps or 0.0
        if qps > best:
            best = qps
            front.append(row)
    return front


def best_recall_rows(rows: list[Row], key: str) -> list[Row]:
    """One row per family at its highest recall (ties -> fastest)."""
    out: dict[str, Row] = {}
    for row in rows:
        value = getattr(row, key)
        if value is None:
            continue
        current = out.get(row.family)
        if current is None:
            out[row.family] = row
            continue
        current_value = getattr(current, key)
        if row.recall > current.recall or (
            row.recall == current.recall and (current_value or 0) > value
        ):
            out[row.family] = row
    return list(out.values())


def subtitle(source: str, extra: str) -> str:
    return f"{extra}  ·  source: {source}"


def finish(fig, path: Path) -> None:
    """Write SVG.

    SVG is the artifact because it is the one that renders identically in a
    browser, in GitHub's markdown viewer and in a PDF build. The PNG encoder in
    this matplotlib build produced a raster whose text layer is unusable when
    re-decoded, so a PNG that looks right on the machine that made it is not a
    safe thing to commit.
    """
    out = path.with_suffix(".svg")
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def scatter_families(ax, rows: list[Row], label_points: bool = False) -> None:
    by_family: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.qps:
            by_family[row.family].append(row)
    for idx, (family, points) in enumerate(sorted(by_family.items())):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        xs = [p.qps for p in points]
        ys = [p.recall for p in points]
        ax.scatter(xs, ys, s=16, color=color, alpha=0.55, linewidths=0)
        front = pareto(points)
        if len(front) > 1:
            ax.plot(
                [p.qps for p in front],
                [p.recall for p in front],
                color=color,
                linewidth=1.4,
                alpha=0.9,
            )
        if label_points:
            peak = max(points, key=lambda p: p.recall)
            ax.annotate(
                family,
                (peak.qps, peak.recall),
                fontsize=7,
                color=color,
                xytext=(3, 3),
                textcoords="offset points",
            )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_pareto(rows: list[Row], dataset: str, out: Path) -> None:
    subset = [r for r in rows if r.dataset == dataset and r.n == 10_000]
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter_families(ax, subset)
    ax.set_xscale("log")
    ax.set_xlabel("queries per second (log)")
    ax.set_ylabel("recall@10")
    ax.set_title(
        subtitle(
            "benchmarks/baseline.jsonl", f"10k points, 96-d, {dataset}, 200 queries"
        ),
        fontsize=9,
        loc="left",
    )
    ax.grid(alpha=0.25)
    handles = [
        plt.Line2D(
            [], [], color=COLOR_CYCLE[i % len(COLOR_CYCLE)], marker="o", linewidth=1.4
        )
        for i, _ in enumerate(sorted({r.family for r in subset}))
    ]
    labels = sorted({r.family for r in subset})
    ax.legend(
        handles,
        labels,
        fontsize=6.5,
        ncol=2,
        loc="lower right",
        frameon=False,
        title="family (line = frontier)",
        title_fontsize=7,
    )
    finish(fig, out)


def fig_recall_vs_ef(rows: list[Row], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, dataset in zip(axes, ("uniform", "clustered"), strict=True):
        subset = [
            r
            for r in rows
            if r.dataset == dataset and r.n == 10_000 and r.knob == "ef_search"
        ]
        by_family: dict[str, list[Row]] = defaultdict(list)
        for row in subset:
            by_family[row.family].append(row)
        for idx, (family, points) in enumerate(sorted(by_family.items())):
            points = sorted(points, key=lambda p: p.knob_value)
            ax.plot(
                [p.knob_value for p in points],
                [p.recall for p in points],
                color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
                marker="o",
                markersize=2.5,
                linewidth=1.1,
                label=family,
            )
        ax.set_xscale("log")
        ax.set_xlabel("ef_search (log)")
        ax.set_title(dataset, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("recall@10")
    axes[0].set_ylim(0, 1.02)
    axes[1].legend(fontsize=6, ncol=2, frameon=False, loc="lower right")
    fig.suptitle(
        subtitle("benchmarks/baseline.jsonl", "10k points, 96-d, 200 queries"),
        fontsize=9,
        x=0.01,
        ha="left",
    )
    finish(fig, out)


def fig_cost_axes(rows: list[Row], out: Path) -> None:
    subset = [r for r in rows if r.n == 10_000 and r.dataset == "clustered"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, label in (
        (axes[0], "build_s", "build seconds"),
        (axes[1], "mem_mib", "resident growth (MiB)"),
    ):
        best = best_recall_rows(subset, key)
        xs, ys, names = [], [], []
        for row in best:
            value = getattr(row, key)
            if value is None:
                continue
            xs.append(value)
            ys.append(row.recall)
            names.append(row.family)
        ax.scatter(xs, ys, s=22, color=COLOR_CYCLE[0], alpha=0.75)
        for x, y, name in zip(xs, ys, names, strict=True):
            ax.annotate(
                name, (x, y), fontsize=6, xytext=(3, 2), textcoords="offset points"
            )
        if key == "build_s":
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("recall@10 at that family's best point")
        ax.grid(alpha=0.25)
    fig.suptitle(
        subtitle(
            "benchmarks/baseline.jsonl",
            "10k clustered: trees pay in latency, graphs in build and memory",
        ),
        fontsize=9,
        x=0.01,
        ha="left",
    )
    finish(fig, out)


def fig_dists(rows: list[Row], out: Path) -> None:
    subset = [r for r in rows if r.dists_per_query]
    fig, ax = plt.subplots(figsize=(8, 5))
    by_family: dict[str, list[Row]] = defaultdict(list)
    for row in subset:
        by_family[row.family].append(row)
    for idx, (family, points) in enumerate(sorted(by_family.items())):
        points = sorted(points, key=lambda p: p.recall)
        ax.plot(
            [p.dists_per_query for p in points],
            [p.recall for p in points],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=family,
        )
    ax.set_xscale("log")
    ax.set_xlabel("distance computations per query (log)")
    ax.set_ylabel("recall@10")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(
        subtitle(
            "benchmarks/large-scale-100k-dists.jsonl",
            "100k clustered: work per query, which wall clock hides",
        ),
        fontsize=9,
        loc="left",
    )
    finish(fig, out)


def fig_tiers(rows: list[Row], out: Path) -> None:
    tiers = sorted(
        {r.n for r in rows if r.dataset == "uniform" and r.n and r.n <= 3_000_000}
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, n in enumerate(tiers):
        points = sorted(
            [r for r in rows if r.n == n and r.dataset == "uniform"],
            key=lambda r: r.knob_value,
        )
        ax.plot(
            [p.knob_value for p in points],
            [p.recall for p in points],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=f"n = {n:,}",
        )
    ax.axhline(0.95, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.annotate("0.95 recall", (ax.get_xlim()[0], 0.955), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("ef_search (log)")
    ax.set_ylabel("recall@10")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        subtitle(
            "benchmarks/large-scale-*.jsonl",
            "uniform 96-d, same index and knob, three tiers",
        ),
        fontsize=9,
        loc="left",
    )
    finish(fig, out)


def fig_tier_cost(rows: list[Row], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key, label, scale in (
        (axes[0], "mem_mib", "index memory (MiB)", "log"),
        (axes[1], "build_s", "build seconds", "log"),
    ):
        seen: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            if row.n is None or getattr(row, key) is None:
                continue
            seen[("hnsw" if row.family == "hnsw" else row.family)].append(row)
        for idx, (family, points) in enumerate(sorted(seen.items())):
            by_n: dict[int, Row] = {}
            for point in points:
                by_n.setdefault(point.n, point)
            xs = sorted(by_n)
            ax.plot(
                xs,
                [getattr(by_n[x], key) for x in xs],
                marker="o",
                color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
                label=family,
                linewidth=1.5,
            )
        ax.set_xscale("log")
        ax.set_yscale(scale)
        ax.set_xlabel("points (log)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        subtitle(
            "benchmarks/large-scale-*.jsonl", "cost of the index as the corpus grows"
        ),
        fontsize=9,
        x=0.01,
        ha="left",
    )
    finish(fig, out)


def fig_storage(rows: list[Row], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for idx, (n, family, label) in enumerate(
        [
            (100_000, "diskann", "100k resident"),
            (100_000, "diskann-file", "100k file"),
            (1_000_000, "diskann", "1M resident"),
            (1_000_000, "diskann-file", "1M file"),
        ]
    ):
        points = sorted(
            [r for r in rows if r.n == n and r.family == family and r.p50_us],
            key=lambda r: r.knob_value,
        )
        ax.plot(
            [p.knob_value for p in points],
            [p.p50_us for p in points],
            marker="o",
            markersize=4,
            linewidth=1.5,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ef_search (log)")
    ax.set_ylabel("p50 latency, µs (log)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("same index, two search paths", fontsize=8, loc="left")

    # Fixed-recall comparison recorded separately (diskann-storage-latency.csv).
    csv_path = ROOT / "docs" / "diskann-storage-latency.csv"
    ax = axes[1]
    if csv_path.exists():
        means: dict[str, list[float]] = defaultdict(list)
        recall: dict[str, float] = {}
        with csv_path.open() as fh:
            for row in csv.DictReader(fh):
                means[row["storage"]].append(float(row["mean_us_per_query"]))
                recall[row["storage"]] = float(row["recall_at_10"])
        names = list(means)
        values = [float(np.median(means[name])) for name in names]
        bars = ax.bar(names, values, color=COLOR_CYCLE[: len(names)], width=0.5)
        for bar, name in zip(bars, names, strict=True):
            ax.annotate(
                f"{values[names.index(name)]:.0f} µs\nrecall {recall[name]:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylabel("median µs per query, matched recall")
        ax.grid(axis="y", alpha=0.25)
        ax.set_title("matched recall, 3 runs each", fontsize=8, loc="left")
        ax.margins(y=0.18)
    else:
        ax.text(0.5, 0.5, "diskann-storage-latency.csv missing", ha="center")
    fig.suptitle(
        subtitle(
            "benchmarks/large-scale-*.jsonl, docs/diskann-storage-latency.csv",
            "storage is a measured axis, not an assumption",
        ),
        fontsize=9,
        x=0.01,
        ha="left",
    )
    finish(fig, out)


def fig_real(name: str, rows: list[Row], out: Path, extra: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    scatter_families(ax, rows, label_points=True)
    ax.set_xscale("log")
    ax.set_xlabel("queries per second (log)")
    ax.set_ylabel("recall@10")
    ax.grid(alpha=0.25)
    ax.set_title(subtitle(f"docs/{name}.jsonl", extra), fontsize=9, loc="left")
    finish(fig, out)


def fig_external_repeats(out: Path) -> None:
    """fashion-mnist 20k: our HNSW against two external libraries, three repeats."""
    path = ROOT / "docs" / "fashion20k-k10.jsonl"
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if not r.get("_meta")]
    by_alg: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_alg[row["algorithm"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    for idx, (alg, points) in enumerate(sorted(by_alg.items())):
        recalls = [p["recall_at_10"] for p in points]
        qps = [p["qps"] for p in points]
        ax.errorbar(
            float(np.mean(qps)),
            float(np.mean(recalls)),
            xerr=float(np.std(qps)),
            yerr=float(np.std(recalls)),
            fmt="o",
            capsize=3,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=f"{alg} (n={len(points)})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("queries per second (log), mean ± sd over repeats")
    ax.set_ylabel("recall@10, mean ± sd")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    names = sorted(by_alg)
    xs = np.arange(len(names))
    heap = [
        float(np.mean([p.get("index_bytes") or 0 for p in by_alg[name]])) / 2**20
        for name in names
    ]
    disk = [
        float(np.mean([p.get("serialized_bytes") or 0 for p in by_alg[name]])) / 2**20
        for name in names
    ]
    ax.bar(xs - 0.2, heap, width=0.38, label="in memory (heap estimate)")
    ax.bar(xs + 0.2, disk, width=0.38, label="serialized")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("MiB")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("footprint per algorithm", fontsize=8, loc="left")
    fig.suptitle(
        subtitle(
            "docs/fashion20k-k10.jsonl",
            "20k x 784, k=10, 3 repeats per algorithm: ours against two external HNSWs",
        ),
        fontsize=9,
        x=0.01,
        ha="left",
    )
    finish(fig, out)


def fig_single_family(out: Path) -> None:
    """One family across the datasets where only it was measured."""
    files = {
        "GloVe-100 (1.18M, 100d)": ROOT / "docs" / "glove-100-angular.jsonl",
        "Fashion-MNIST (60k, 784d)": (
            ROOT / "docs" / "fashion-mnist-784-euclidean.jsonl"
        ),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for idx, (label, path) in enumerate(files.items()):
        if not path.exists():
            continue
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        rows = [
            r for r in rows if not r.get("_meta") and r.get("recall_at_10") is not None
        ]
        if not rows:
            continue
        rows.sort(key=lambda r: r["recall_at_10"])
        ax.plot(
            [r["qps"] for r in rows],
            [r["recall_at_10"] for r in rows],
            marker="o",
            linewidth=1.5,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=label,
        )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("queries per second (log)")
    ax.set_ylabel("recall@10")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        subtitle(
            "docs/glove-100-angular.jsonl, docs/fashion-mnist-784-euclidean.jsonl",
            "SQ8U, the only family measured on these two datasets",
        ),
        fontsize=9,
        loc="left",
    )
    finish(fig, out)


def fig_layout(out: Path) -> None:
    """Same graph, two resident layouts."""
    files = {
        "default layout": ROOT / "benchmarks" / "large-scale-1m-hnsw-layout.jsonl",
        "upper layers frozen": (
            ROOT / "benchmarks" / "large-scale-1m-hnsw-compact.jsonl"
        ),
    }
    loaded: dict[str, list[dict]] = {}
    for label, path in files.items():
        if not path.exists():
            return
        loaded[label] = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    names = list(loaded)
    mem = [
        next(r for r in loaded[name] if not r.get("_meta"))["mem_mib"] for name in names
    ]
    bars = ax.bar(names, mem, color=COLOR_CYCLE[: len(names)], width=0.5)
    for bar, value in zip(bars, mem, strict=True):
        ax.annotate(
            f"{value:.0f} MiB",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("index memory (MiB), 1M points")
    ax.set_ylim(0, max(mem) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("resident size", fontsize=8, loc="left")

    ax = axes[1]
    for idx, (name, rows) in enumerate(loaded.items()):
        points = sorted(
            (r for r in rows if not r.get("_meta")),
            key=lambda r: r["knob_value"],
        )
        ax.plot(
            [p["knob_value"] for p in points],
            [p["p50_us"] for p in points],
            marker="o",
            linewidth=1.5,
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
            label=name,
        )
    ax.set_xscale("log")
    ax.set_xlabel("ef_search (log)")
    ax.set_ylabel("p50 latency, µs")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("query cost is unchanged", fontsize=8, loc="left")
    fig.suptitle(
        subtitle(
            "benchmarks/large-scale-1m-hnsw-{layout,compact}.jsonl",
            "1M clustered: freezing upper layers into eligible-node CSR",
        ),
        fontsize=9,
        x=0.01,
        ha="left",
    )
    finish(fig, out)


def fig_seeds(out: Path) -> None:
    path = ROOT / "docs" / "hnsw-seed-recall.csv"
    if not path.exists():
        return
    per_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with path.open() as fh:
        for row in csv.DictReader(fh):
            per_seed[row["seed"]][int(row["ef_search"])].append(
                float(row["recall_at_10"])
            )
    efs = sorted({ef for seeds in per_seed.values() for ef in seeds})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.8 / max(len(per_seed), 1)
    for idx, (seed, by_ef) in enumerate(sorted(per_seed.items())):
        xs = [i + idx * width for i, _ in enumerate(efs)]
        ys = [float(np.mean(by_ef.get(ef, [np.nan]))) for ef in efs]
        err = [
            float(np.std(by_ef.get(ef, [0.0]))) if len(by_ef.get(ef, [])) > 1 else 0.0
            for ef in efs
        ]
        ax.bar(
            xs,
            ys,
            width=width,
            yerr=err,
            capsize=2,
            label=f"seed {seed}",
            color=COLOR_CYCLE[idx % len(COLOR_CYCLE)],
        )
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(efs))])
    ax.set_xticklabels([str(ef) for ef in efs])
    ax.set_xlabel("ef_search")
    ax.set_ylabel("recall@10 (mean, ± sd within seed)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title(
        subtitle(
            "docs/hnsw-seed-recall.csv",
            "seed variance is part of the measurement, not noise to hide",
        ),
        fontsize=9,
        loc="left",
    )
    finish(fig, out)


def fig_coverage(rows: list[Row], out: Path) -> None:
    metrics = [
        "recall",
        "qps",
        "p50_us",
        "p99_us",
        "build_s",
        "mem_mib",
        "dists_per_query",
    ]
    families = sorted({r.family for r in rows if r.n == 10_000})
    grid = np.zeros((len(families), len(metrics)))
    for i, family in enumerate(families):
        subset = [r for r in rows if r.family == family and r.n == 10_000]
        for j, metric in enumerate(metrics):
            grid[i, j] = sum(1 for r in subset if getattr(r, metric) is not None)
    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    im = ax.imshow(grid, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=7)
    for i in range(len(families)):
        for j in range(len(metrics)):
            if grid[i, j] == 0:
                ax.text(
                    j, i, "–", ha="center", va="center", fontsize=7, color="#999999"
                )
    fig.colorbar(im, ax=ax, label="rows carrying the metric", shrink=0.6)
    ax.set_title(
        subtitle("benchmarks/baseline.jsonl", "what each family actually has measured"),
        fontsize=9,
        loc="left",
    )
    finish(fig, out)


def fig_competitor_parity(parity: dict[str, list[Row]], out: Path) -> None:
    """Recall against latency for hnswlib, USearch and vicinity on one protocol.

    Both libraries saw the same vectors, M, ef_construction, k and query set, and
    ground truth is an exact scan on each side. The x axis is p50 latency rather
    than QPS because the rows come from one run per point on one host; a QPS axis
    would invite reading throughput differences that are scheduler noise.
    """
    tiers = [name for name in parity if name.startswith("hnswlib-parity-")]
    if not tiers:
        return
    fig, axes = plt.subplots(1, len(tiers), figsize=(11, 4.6))
    if len(tiers) == 1:
        axes = [axes]
    colors = {"hnsw": "#1f77b4", "hnswlib": "#d55e00", "usearch": "#009e73"}
    labels = {"hnsw": "vicinity hnsw", "hnswlib": "hnswlib", "usearch": "USearch"}
    for ax, name in zip(axes, sorted(tiers), strict=True):
        rows = [r for r in parity[name] if r.p50_us]
        for family in sorted({r.family for r in rows}):
            points = sorted(
                (r for r in rows if r.family == family), key=lambda r: r.p50_us
            )
            ax.plot(
                [p.p50_us for p in points],
                [p.recall for p in points],
                color=colors.get(family, "#7f7f7f"),
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=labels.get(family, family),
            )
            for point in points:
                ax.annotate(
                    f"{point.knob_value:.0f}",
                    (point.p50_us, point.recall),
                    fontsize=6.5,
                    xytext=(3, -6),
                    textcoords="offset points",
                    color=colors.get(family, "#7f7f7f"),
                )
        tier = name.rsplit("-", 1)[-1].removesuffix(".jsonl")
        ax.set_xscale("log")
        ax.set_xlabel("p50 latency, µs (log)")
        ax.set_ylabel("recall@10")
        ax.set_title(
            subtitle(f"benchmarks/{name}", f"clustered 96-d, {tier}, one thread"),
            fontsize=9,
            loc="left",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    finish(fig, out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    baseline = data.get("baseline.jsonl", [])
    large = [
        row
        for name, rows in data.items()
        if name.startswith("large-scale")
        for row in rows
    ]

    fig_pareto(baseline, "uniform", OUT / "pareto-uniform-10k.png")
    fig_pareto(baseline, "clustered", OUT / "pareto-clustered-10k.png")
    fig_recall_vs_ef(baseline, OUT / "recall-vs-ef-10k.png")
    fig_cost_axes(baseline, OUT / "cost-axes-10k.png")
    fig_dists(large, OUT / "dists-vs-recall-100k.png")
    fig_tiers(large, OUT / "tiers-recall-vs-ef-uniform.png")
    fig_tier_cost(large, OUT / "tier-memory-and-build.png")
    fig_storage(large, OUT / "storage-resident-vs-file.png")
    fig_coverage(baseline, OUT / "metric-coverage-10k.png")

    for name, extra in (
        ("glove-25-angular", "GloVe-25, 1.18M vectors, angular, 10k queries"),
        ("sift-128-euclidean", "SIFT-128, 1M vectors, euclidean"),
        ("gist-960-euclidean", "GIST-960, 1M vectors, euclidean"),
    ):
        rows = data.get(f"{name}.jsonl", [])
        if rows:
            fig_real(name, rows, OUT / f"real-{name}.png", extra)
    fig_external_repeats(OUT / "external-repeats-fashion20k.svg")
    fig_single_family(OUT / "single-family-datasets.svg")
    fig_layout(OUT / "layout-upper-layers.svg")
    fig_seeds(OUT / "seed-variance.png")
    fig_competitor_parity(
        {
            name: rows
            for name, rows in data.items()
            if name.startswith("hnswlib-parity-")
        },
        OUT / "competitor-parity-hnswlib.png",
    )


if __name__ == "__main__":
    main()
