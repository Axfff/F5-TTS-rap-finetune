import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_metrics_results(csv_path):
    """Load objective metrics results from CSV"""
    df = pd.read_csv(csv_path)
    return df


def extract_metric_pairs(df):
    """Extract baseline/finetuned pairs for each metric"""
    metric_pairs = {}

    for col in df.columns:
        if col.startswith("baseline_"):
            metric_name = col.replace("baseline_", "")
            finetuned_col = f"finetuned_{metric_name}"

            if finetuned_col in df.columns:
                baseline_vals = df[col].dropna()
                finetuned_vals = df[finetuned_col].dropna()

                if len(baseline_vals) > 0 and len(finetuned_vals) > 0:
                    metric_pairs[metric_name] = {
                        "baseline": baseline_vals.values,
                        "finetuned": finetuned_vals.values,
                        "baseline_mean": baseline_vals.mean(),
                        "finetuned_mean": finetuned_vals.mean(),
                        "baseline_std": baseline_vals.std(),
                        "finetuned_std": finetuned_vals.std(),
                    }

    return metric_pairs


def determine_metric_direction(metric_name):
    """Determine if higher or lower is better for a metric"""
    higher_is_better = ["PESQ", "STOI", "SISDR", "FWSEGSNR", "SSNR", "SNR", "SRMR", "CSIG", "CBAK", "COVL"]

    lower_is_better = ["MCD", "LSD", "LLR"]

    for metric in higher_is_better:
        if metric in metric_name.upper():
            return "higher"

    for metric in lower_is_better:
        if metric in metric_name.upper():
            return "lower"

    if "DNSMOS" in metric_name.upper() or "MOS" in metric_name.upper():
        return "higher"

    return "higher"


def plot_metric_comparison_bars(metric_pairs, output_dir):
    """Plot side-by-side bar comparison for each metric"""
    n_metrics = len(metric_pairs)
    if n_metrics == 0:
        print("No metrics to plot")
        return

    fig, axes = plt.subplots(
        (n_metrics + 2) // 3, 3, figsize=(18, 6 * ((n_metrics + 2) // 3)), squeeze=False
    )
    axes = axes.flatten()

    for idx, (metric_name, data) in enumerate(metric_pairs.items()):
        ax = axes[idx]

        baseline_mean = data["baseline_mean"]
        finetuned_mean = data["finetuned_mean"]
        baseline_std = data["baseline_std"]
        finetuned_std = data["finetuned_std"]

        direction = determine_metric_direction(metric_name)
        if direction == "higher":
            better_color = "#2ca02c" if finetuned_mean > baseline_mean else "#ff7f0e"
        else:
            better_color = "#2ca02c" if finetuned_mean < baseline_mean else "#ff7f0e"

        x = np.arange(2)
        means = [baseline_mean, finetuned_mean]
        stds = [baseline_std, finetuned_std]
        colors = ["#ff7f0e", better_color]

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5, edgecolor="black")

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Finetuned"], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"{metric_name}\n({'Higher' if direction == 'higher' else 'Lower'} is better)", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison_bars.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'metric_comparison_bars.png'}")


def plot_radar_chart(metric_pairs, output_dir):
    """Plot radar chart comparing baseline vs finetuned across all metrics"""
    if len(metric_pairs) == 0:
        print("No metrics to plot radar chart")
        return

    metrics = list(metric_pairs.keys())
    n_metrics = len(metrics)

    baseline_scores = []
    finetuned_scores = []

    for metric_name in metrics:
        data = metric_pairs[metric_name]
        baseline_mean = data["baseline_mean"]
        finetuned_mean = data["finetuned_mean"]

        direction = determine_metric_direction(metric_name)

        all_values = list(data["baseline"]) + list(data["finetuned"])
        min_val = min(all_values)
        max_val = max(all_values)

        if max_val - min_val > 0:
            baseline_norm = (baseline_mean - min_val) / (max_val - min_val)
            finetuned_norm = (finetuned_mean - min_val) / (max_val - min_val)

            if direction == "lower":
                baseline_norm = 1 - baseline_norm
                finetuned_norm = 1 - finetuned_norm
        else:
            baseline_norm = 0.5
            finetuned_norm = 0.5

        baseline_scores.append(baseline_norm)
        finetuned_scores.append(finetuned_norm)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    baseline_scores += baseline_scores[:1]
    finetuned_scores += finetuned_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    ax.plot(angles, baseline_scores, "o-", linewidth=2, label="Baseline", color="#ff7f0e")
    ax.fill(angles, baseline_scores, alpha=0.25, color="#ff7f0e")

    ax.plot(angles, finetuned_scores, "o-", linewidth=2, label="Finetuned", color="#2ca02c")
    ax.fill(angles, finetuned_scores, alpha=0.25, color="#2ca02c")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_title("Multi-Metric Comparison (Normalized)\nHigher is Better", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'radar_chart.png'}")


def plot_box_plots(metric_pairs, output_dir):
    """Plot box plots showing distribution of scores"""
    if len(metric_pairs) == 0:
        print("No metrics to plot box plots")
        return

    n_metrics = len(metric_pairs)
    fig, axes = plt.subplots((n_metrics + 2) // 3, 3, figsize=(18, 6 * ((n_metrics + 2) // 3)), squeeze=False)
    axes = axes.flatten()

    for idx, (metric_name, data) in enumerate(metric_pairs.items()):
        ax = axes[idx]

        box_data = [data["baseline"], data["finetuned"]]
        positions = [1, 2]

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
        )

        colors = ["#ff7f0e", "#2ca02c"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(["Baseline", "Finetuned"], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"{metric_name} Distribution", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / "box_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'box_plots.png'}")


def plot_scatter_comparison(metric_pairs, output_dir):
    """Plot scatter plots comparing baseline vs finetuned scores"""
    if len(metric_pairs) == 0:
        print("No metrics to plot scatter comparison")
        return

    n_metrics = len(metric_pairs)
    fig, axes = plt.subplots((n_metrics + 2) // 3, 3, figsize=(18, 6 * ((n_metrics + 2) // 3)), squeeze=False)
    axes = axes.flatten()

    for idx, (metric_name, data) in enumerate(metric_pairs.items()):
        ax = axes[idx]

        baseline = data["baseline"]
        finetuned = data["finetuned"]

        min_len = min(len(baseline), len(finetuned))
        baseline = baseline[:min_len]
        finetuned = finetuned[:min_len]

        ax.scatter(baseline, finetuned, alpha=0.6, s=50, edgecolors="black", linewidths=0.5)

        all_vals = np.concatenate([baseline, finetuned])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = (max_val - min_val) * 0.1
        ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], "r--", linewidth=2, label="y=x")

        ax.set_xlabel("Baseline Score", fontsize=11)
        ax.set_ylabel("Finetuned Score", fontsize=11)
        ax.set_title(f"{metric_name}: Baseline vs Finetuned", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=9)

        direction = determine_metric_direction(metric_name)
        if direction == "higher":
            better_region = "upper left"
        else:
            better_region = "lower right"
        ax.text(
            0.05,
            0.95,
            f"Better: {better_region}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / "scatter_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'scatter_comparison.png'}")


def plot_win_rate(metric_pairs, output_dir):
    """Plot win rate: how many times finetuned beats baseline"""
    if len(metric_pairs) == 0:
        print("No metrics to plot win rate")
        return

    metrics = []
    win_rates = []
    colors_list = []

    for metric_name, data in metric_pairs.items():
        baseline = data["baseline"]
        finetuned = data["finetuned"]

        min_len = min(len(baseline), len(finetuned))
        baseline = baseline[:min_len]
        finetuned = finetuned[:min_len]

        direction = determine_metric_direction(metric_name)

        if direction == "higher":
            wins = np.sum(finetuned > baseline)
        else:
            wins = np.sum(finetuned < baseline)

        win_rate = (wins / min_len) * 100

        metrics.append(metric_name)
        win_rates.append(win_rate)
        colors_list.append("#2ca02c" if win_rate > 50 else "#ff7f0e")

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(metrics, win_rates, color=colors_list, alpha=0.8, edgecolor="black")

    for bar, rate in zip(bars, win_rates):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2.0, f"{rate:.1f}%", ha="left", va="center", fontsize=10, fontweight="bold")

    ax.axvline(x=50, color="red", linestyle="--", linewidth=2, label="50% (Equal)")
    ax.set_xlabel("Win Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title("Finetuned Win Rate vs Baseline\n(% of samples where finetuned is better)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "win_rate.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'win_rate.png'}")


def plot_heatmap(df, metric_pairs, output_dir):
    """Plot heatmap showing per-sample metric comparison"""
    if len(metric_pairs) == 0:
        print("No metrics to plot heatmap")
        return

    improvement_data = []
    metric_names = []

    for metric_name, data in metric_pairs.items():
        baseline = data["baseline"]
        finetuned = data["finetuned"]

        min_len = min(len(baseline), len(finetuned))
        baseline = baseline[:min_len]
        finetuned = finetuned[:min_len]

        direction = determine_metric_direction(metric_name)

        if direction == "higher":
            improvement = finetuned - baseline
        else:
            improvement = baseline - finetuned

        improvement_data.append(improvement)
        metric_names.append(metric_name)

    if len(improvement_data) == 0:
        print("No improvement data to plot")
        return

    improvement_matrix = np.array(improvement_data)

    fig, ax = plt.subplots(figsize=(max(12, min_len * 0.5), max(8, len(metric_names) * 0.5)))

    im = ax.imshow(improvement_matrix, cmap="RdYlGn", aspect="auto")

    ax.set_xticks(np.arange(min_len))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels([f"S{i+1}" for i in range(min_len)], fontsize=8)
    ax.set_yticklabels(metric_names, fontsize=10)

    ax.set_xlabel("Sample", fontsize=12, fontweight="bold")
    ax.set_ylabel("Metric", fontsize=12, fontweight="bold")
    ax.set_title("Per-Sample Improvement Heatmap\n(Green = Finetuned Better, Red = Baseline Better)", fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Improvement", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_dir / 'heatmap.png'}")


def generate_summary_stats(metric_pairs, output_dir):
    """Generate summary statistics text file"""
    summary = """
Objective Metrics Evaluation Summary
=====================================

"""

    for metric_name, data in metric_pairs.items():
        baseline_mean = data["baseline_mean"]
        finetuned_mean = data["finetuned_mean"]
        baseline_std = data["baseline_std"]
        finetuned_std = data["finetuned_std"]

        direction = determine_metric_direction(metric_name)

        if direction == "higher":
            improvement = finetuned_mean - baseline_mean
            improvement_pct = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0
            better = "Finetuned" if improvement > 0 else "Baseline"
        else:
            improvement = baseline_mean - finetuned_mean
            improvement_pct = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0
            better = "Finetuned" if improvement > 0 else "Baseline"

        summary += f"{metric_name} ({'Higher' if direction == 'higher' else 'Lower'} is better):\n"
        summary += f"  Baseline:  {baseline_mean:.4f} ± {baseline_std:.4f}\n"
        summary += f"  Finetuned: {finetuned_mean:.4f} ± {finetuned_std:.4f}\n"
        summary += f"  Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)\n"
        summary += f"  Winner: {better}\n\n"

    baseline = data["baseline"]
    finetuned = data["finetuned"]
    min_len = min(len(baseline), len(finetuned))

    summary += "\nOverall Win Rate:\n"
    summary += "-" * 40 + "\n"

    total_wins = 0
    total_comparisons = 0

    for metric_name, data in metric_pairs.items():
        baseline = data["baseline"]
        finetuned = data["finetuned"]

        min_len = min(len(baseline), len(finetuned))
        baseline = baseline[:min_len]
        finetuned = finetuned[:min_len]

        direction = determine_metric_direction(metric_name)

        if direction == "higher":
            wins = np.sum(finetuned > baseline)
        else:
            wins = np.sum(finetuned < baseline)

        win_rate = (wins / min_len) * 100
        summary += f"{metric_name:20s}: {win_rate:5.1f}% ({wins}/{min_len})\n"

        total_wins += wins
        total_comparisons += min_len

    overall_win_rate = (total_wins / total_comparisons * 100) if total_comparisons > 0 else 0
    summary += f"\n{'Overall':20s}: {overall_win_rate:5.1f}% ({total_wins}/{total_comparisons})\n"

    summary_path = output_dir / "summary_stats.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"\nSaved: {summary_path}")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Visualize objective metrics evaluation results")

    parser.add_argument("--csv", type=str, required=True, help="Path to objective metrics CSV file")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save visualizations (default: same as CSV)"
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent / "visualizations_objective"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading objective metrics results from: {csv_path}")
    df = load_metrics_results(csv_path)

    print(f"Found {len(df)} samples")

    metric_pairs = extract_metric_pairs(df)
    print(f"Found {len(metric_pairs)} metrics to visualize")

    print(f"\nGenerating visualizations in: {output_dir}\n")

    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    plot_metric_comparison_bars(metric_pairs, output_dir)
    plot_radar_chart(metric_pairs, output_dir)
    plot_box_plots(metric_pairs, output_dir)
    plot_scatter_comparison(metric_pairs, output_dir)
    plot_win_rate(metric_pairs, output_dir)
    plot_heatmap(df, metric_pairs, output_dir)
    generate_summary_stats(metric_pairs, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()


"""example usage

python src/f5_tts/eval/eval_visualize_objective_metrics.py \
    --csv results/rap_evaluation/objective_metrics.csv \
    --output_dir results/rap_evaluation/visualizations_objective

"""
