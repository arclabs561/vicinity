#!/usr/bin/env python3
"""Generate Pareto frontier visualizations from benchmark results.

Based on ann-benchmarks visualization style:
- Recall vs QPS scatter with confidence bands
- LID-stratified recall comparison
- Multi-scale comparison

Run: uvx --with numpy --with matplotlib python scripts/plot_pareto.py
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pareto_frontier(points: list[dict]) -> list[dict]:
    """Keep nondominated recall/QPS records, ordered by increasing recall.

    A higher-recall record at equal QPS dominates the lower-recall record, and
    a higher-QPS record at equal recall dominates the lower-QPS record. Exact
    ties retain the lower ``ef`` record so the plotted error bars are stable.
    """
    ordered = sorted(
        points,
        key=lambda point: (
            -float(point["recall_mean"]),
            -float(point["qps_mean"]),
            int(point["ef"]),
        ),
    )
    frontier = []
    best_qps = float("-inf")
    for point in ordered:
        qps = float(point["qps_mean"])
        if qps > best_qps:
            frontier.append(point)
            best_qps = qps
    return sorted(frontier, key=lambda point: float(point["recall_mean"]))


def fastest_at_recall(points: list[dict], recall_floor: float) -> float:
    """Return the fastest measured QPS meeting a recall floor, or NaN."""
    qualifying = [
        float(point["qps_mean"])
        for point in points
        if float(point["recall_mean"]) >= recall_floor
    ]
    return max(qualifying, default=float("nan"))


def load_results(data_dir: Path) -> dict:
    """Load all benchmark results."""
    results = {}
    # Check all possible scales
    for scale in ["S", "M", "L", "XL", "B", "T", "P"]:
        result_file = data_dir / f"results_{scale}.json"
        if result_file.exists():
            with open(result_file) as f:
                results[scale] = json.load(f)
    return results


def plot_pareto_frontier(results: dict, output_dir: Path):
    """Plot Recall vs QPS Pareto frontier with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors and markers for all possible scales
    colors = {
        "S": "#1f77b4", "M": "#ff7f0e", "L": "#2ca02c", "XL": "#d62728",
        "B": "#1f77b4", "T": "#ff7f0e", "P": "#2ca02c",  # Legacy
    }
    markers = {
        "S": "o", "M": "s", "L": "^", "XL": "D",
        "B": "o", "T": "s", "P": "^",  # Legacy
    }
    
    for scale, data in results.items():
        points = pareto_frontier(data["pareto_points"])
        
        recalls = [p["recall_mean"] * 100 for p in points]
        qps = [p["qps_mean"] for p in points]
        recall_ci_low = [p["recall_ci_low"] * 100 for p in points]
        recall_ci_high = [p["recall_ci_high"] * 100 for p in points]
        
        n_train = data["n_train"]
        label = f"Scale {scale} ({n_train:,} vectors)"
        
        # Plot points with error bars
        ax.errorbar(
            recalls, qps,
            xerr=[
                [r - l for r, l in zip(recalls, recall_ci_low)],
                [h - r for r, h in zip(recalls, recall_ci_high)]
            ],
            fmt=markers[scale],
            color=colors[scale],
            label=label,
            capsize=3,
            markersize=8,
            linewidth=1.5,
        )
        
        # Connect only the non-dominated frontier, from lower to higher recall.
        ax.plot(
            recalls,
            qps,
            color=colors[scale],
            linestyle="--",
            alpha=0.5,
        )
        
        # Add brute-force baseline
        bf_qps = data["brute_force_qps"]
        ax.axhline(
            y=bf_qps,
            color=colors[scale],
            linestyle=":",
            alpha=0.3,
            label=f"Brute-force {scale} ({bf_qps:.0f} QPS)",
        )
    
    ax.set_xlabel("Recall@10 (%)", fontsize=12)
    ax.set_ylabel("Queries per Second (QPS)", fontsize=12)
    ax.set_title("HNSW Recall-QPS Pareto Frontier (95% CI)", fontsize=14)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.png", dpi=150)
    plt.savefig(output_dir / "pareto_frontier.svg")
    print(f"Saved: {output_dir / 'pareto_frontier.png'}")
    plt.close()


def plot_lid_stratified(results: dict, output_dir: Path):
    """Plot recall by query difficulty (LID-stratified)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5), sharey=True)
    
    if len(results) == 1:
        axes = [axes]
    
    for ax, (scale, data) in zip(axes, results.items()):
        points = data["pareto_points"]
        efs = [p["ef"] for p in points]
        
        easy = [p["recall_easy"] * 100 for p in points]
        medium = [p["recall_medium"] * 100 for p in points]
        hard = [p["recall_hard"] * 100 for p in points]
        
        x = np.arange(len(efs))
        width = 0.25
        
        ax.bar(x - width, easy, width, label="Easy (low LID)", color="#2ecc71")
        ax.bar(x, medium, width, label="Medium", color="#f39c12")
        ax.bar(x + width, hard, width, label="Hard (high LID)", color="#e74c3c")
        
        ax.set_xlabel("ef (search parameter)")
        ax.set_ylabel("Recall@10 (%)")
        ax.set_title(f"Scale {scale}: {data['n_train']:,} vectors")
        ax.set_xticks(x)
        ax.set_xticklabels(efs)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.suptitle("Recall by Query Difficulty (LID-stratified)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "lid_stratified_recall.png", dpi=150)
    print(f"Saved: {output_dir / 'lid_stratified_recall.png'}")
    plt.close()


def plot_scaling_behavior(results: dict, output_dir: Path):
    """Plot how metrics scale with dataset size."""
    if len(results) < 2:
        print("Need at least 2 scales for scaling plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    scales = sorted(results, key=lambda scale: results[scale]["n_train"])
    n_trains = [results[s]["n_train"] for s in scales]
    
    # Build time scaling
    ax = axes[0, 0]
    build_times = [results[s]["build_time_s"] for s in scales]
    ax.plot(n_trains, build_times, "o-", markersize=10, linewidth=2)
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Build Time (s)")
    ax.set_title("Index Build Time Scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Fastest measured setting at 90% recall. A missing target stays visibly
    # absent instead of substituting a lower-recall setting.
    ax = axes[0, 1]
    qps_90 = [
        fastest_at_recall(results[scale]["pareto_points"], 0.90)
        for scale in scales
    ]
    ax.plot(n_trains, qps_90, "o-", markersize=10, linewidth=2, color="green")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("QPS at Recall ≥90%")
    ax.set_title("Throughput at Target Recall")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Recall variance
    ax = axes[1, 0]
    for s in scales:
        points = results[s]["pareto_points"]
        efs = [p["ef"] for p in points]
        stds = [p["recall_std"] * 100 for p in points]
        ax.plot(efs, stds, "o-", label=f"Scale {s}", markersize=8)
    ax.set_xlabel("ef (search parameter)")
    ax.set_ylabel("Recall Std Dev (%)")
    ax.set_title("Recall Variance Across Runs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # p99 latency
    ax = axes[1, 1]
    for s in scales:
        points = results[s]["pareto_points"]
        recalls = [p["recall_mean"] * 100 for p in points]
        p99s = [p["latency_p99_us"] for p in points]
        ax.plot(recalls, p99s, "o-", label=f"Scale {s}", markersize=8)
    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("p99 Latency (us)")
    ax.set_title("Tail Latency vs Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Scaling Behavior Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "scaling_analysis.png", dpi=150)
    print(f"Saved: {output_dir / 'scaling_analysis.png'}")
    plt.close()


def generate_html_report(results: dict, output_dir: Path):
    """Generate an HTML report with all plots and metrics."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Jin ANN Benchmark Report</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
        th { background: #f5f5f5; }
        .metric-good { color: #27ae60; }
        .metric-warn { color: #f39c12; }
        .metric-bad { color: #e74c3c; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
        .summary { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Jin ANN Benchmark Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Benchmark methodology based on <a href="https://ann-benchmarks.com">ann-benchmarks</a> 
        and <a href="https://big-ann-benchmarks.com">BigANN NeurIPS 2023</a> best practices.</p>
        <ul>
            <li>Statistical rigor: 5 runs per configuration, 95% confidence intervals</li>
            <li>LID-stratified evaluation: Easy/Medium/Hard query difficulty</li>
            <li>Brute-force baseline for speedup calculation</li>
        </ul>
    </div>
    
    <h2>Pareto Frontier</h2>
    <p>Recall@10 vs Queries per Second trade-off with 95% confidence intervals.</p>
    <img src="pareto_frontier.png" alt="Pareto Frontier">
    
    <h2>Query Difficulty Analysis</h2>
    <p>Recall stratified by Local Intrinsic Dimensionality (LID). 
    Hard queries (high LID) are in sparse regions of the embedding space.</p>
    <img src="lid_stratified_recall.png" alt="LID Stratified Recall">
    
    <h2>Scaling Analysis</h2>
    <img src="scaling_analysis.png" alt="Scaling Analysis">
    
    <h2>Detailed Results</h2>
"""
    
    for scale, data in results.items():
        html += f"""
    <h3>Scale {scale}: {data['n_train']:,} vectors x {data['dim']} dims</h3>
    <ul>
        <li>Build time: {data['build_time_s']:.2f}s</li>
        <li>Brute-force baseline: {data['brute_force_qps']:.1f} QPS</li>
    </ul>
    <table>
        <tr>
            <th>ef</th>
            <th>Recall</th>
            <th>95% CI</th>
            <th>QPS</th>
            <th>p99 Latency</th>
            <th>Easy</th>
            <th>Medium</th>
            <th>Hard</th>
        </tr>
"""
        for p in data["pareto_points"]:
            recall_class = "metric-good" if p["recall_mean"] >= 0.95 else ("metric-warn" if p["recall_mean"] >= 0.90 else "")
            html += f"""        <tr>
            <td>{p['ef']}</td>
            <td class="{recall_class}">{p['recall_mean']*100:.1f}%</td>
            <td>[{p['recall_ci_low']*100:.1f}, {p['recall_ci_high']*100:.1f}]</td>
            <td>{p['qps_mean']:.0f}</td>
            <td>{p['latency_p99_us']:.0f}us</td>
            <td>{p['recall_easy']*100:.1f}%</td>
            <td>{p['recall_medium']*100:.1f}%</td>
            <td>{p['recall_hard']*100:.1f}%</td>
        </tr>
"""
        html += "    </table>\n"
    
    html += """
    <h2>Methodology Notes</h2>
    <ul>
        <li><strong>Recall@10</strong>: Fraction of true 10-nearest neighbors found</li>
        <li><strong>QPS</strong>: Queries per second (higher is better)</li>
        <li><strong>95% CI</strong>: Confidence interval from 5 independent runs</li>
        <li><strong>LID Stratification</strong>: Queries partitioned by local intrinsic dimensionality
            <ul>
                <li>Easy: Dense regions, low LID (33rd percentile)</li>
                <li>Medium: Typical queries (33rd-66th percentile)</li>
                <li>Hard: Sparse regions, high LID (66th+ percentile)</li>
            </ul>
        </li>
    </ul>
    
    <p><em>Generated by vicinity benchmark suite. 
    Based on <a href="https://neurips.cc/public/guides/PaperChecklist">NeurIPS paper guidelines</a> 
    for statistical rigor.</em></p>
</body>
</html>
"""
    
    report_path = output_dir / "benchmark_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Saved: {report_path}")


def main():
    # Find data directory
    data_dir = Path(__file__).parent.parent / "data" / "multiscale"
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Run benchmark first: cargo run --example 04_rigorous_benchmark --release")
        sys.exit(1)
    
    output_dir = data_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    
    print("Loading benchmark results...")
    results = load_results(data_dir)
    
    if not results:
        print("No benchmark results found. Run benchmarks first.")
        sys.exit(1)
    
    print(f"Found results for scales: {list(results.keys())}")
    
    print("\nGenerating plots...")
    plot_pareto_frontier(results, output_dir)
    plot_lid_stratified(results, output_dir)
    
    if len(results) >= 2:
        plot_scaling_behavior(results, output_dir)
    
    print("\nGenerating HTML report...")
    generate_html_report(results, output_dir)
    
    print(f"\nAll outputs in: {output_dir}")


if __name__ == "__main__":
    main()
