import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_evaluation_results(csv_path):
    """Load evaluation results from CSV"""
    df = pd.read_csv(csv_path)

    # Map user choices to actual model preferences
    def get_actual_choice(row):
        if row['choice'] == 'equal':
            return 'equal'
        elif row['choice'] == 'a':
            return row['a_is']
        elif row['choice'] == 'b':
            return row['b_is']
        else:
            return 'unknown'

    df['actual_choice'] = df.apply(get_actual_choice, axis=1)

    return df


def plot_overall_preference(df, output_dir):
    """Plot overall model preference"""
    fig, ax = plt.subplots(figsize=(10, 6))

    choice_counts = df['actual_choice'].value_counts()

    colors = {'baseline': '#ff7f0e', 'finetuned': '#2ca02c', 'equal': '#7f7f7f'}
    bar_colors = [colors.get(choice, '#1f77b4') for choice in choice_counts.index]

    bars = ax.bar(choice_counts.index, choice_counts.values, color=bar_colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Model Preference', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Choices', fontsize=14, fontweight='bold')
    ax.set_title('Overall Model Preference', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'overall_preference.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'overall_preference.png'}")


def plot_preference_percentage(df, output_dir):
    """Plot preference as percentage pie chart"""
    fig, ax = plt.subplots(figsize=(10, 8))

    choice_counts = df['actual_choice'].value_counts()

    colors = {'baseline': '#ff7f0e', 'finetuned': '#2ca02c', 'equal': '#7f7f7f'}
    pie_colors = [colors.get(choice, '#1f77b4') for choice in choice_counts.index]

    wedges, texts, autotexts = ax.pie(
        choice_counts.values,
        labels=choice_counts.index,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')

    ax.set_title('Model Preference Distribution', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / 'preference_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'preference_percentage.png'}")


def plot_win_rate(df, output_dir):
    """Plot win rate comparison (excluding equal choices)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Exclude equal choices for win rate calculation
    df_no_equal = df[df['actual_choice'] != 'equal']

    if len(df_no_equal) == 0:
        print("No non-equal choices found, skipping win rate plot")
        return

    choice_counts = df_no_equal['actual_choice'].value_counts()
    total = choice_counts.sum()

    win_rates = (choice_counts / total * 100).sort_values(ascending=False)

    colors = {'baseline': '#ff7f0e', 'finetuned': '#2ca02c'}
    bar_colors = [colors.get(choice, '#1f77b4') for choice in win_rates.index]

    bars = ax.bar(win_rates.index, win_rates.values, color=bar_colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Win Rate Comparison (Excluding Equal Choices)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'win_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'win_rate.png'}")


def plot_timeline(df, output_dir):
    """Plot preference over time"""
    fig, ax = plt.subplots(figsize=(12, 6))

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Create cumulative counts
    df['sample_num'] = range(1, len(df) + 1)

    baseline_cumsum = (df['actual_choice'] == 'baseline').cumsum()
    finetuned_cumsum = (df['actual_choice'] == 'finetuned').cumsum()
    equal_cumsum = (df['actual_choice'] == 'equal').cumsum()

    ax.plot(df['sample_num'], baseline_cumsum, label='Baseline', color='#ff7f0e', linewidth=2, marker='o', markersize=4)
    ax.plot(df['sample_num'], finetuned_cumsum, label='Finetuned', color='#2ca02c', linewidth=2, marker='s', markersize=4)
    ax.plot(df['sample_num'], equal_cumsum, label='Equal', color='#7f7f7f', linewidth=2, marker='^', markersize=4)

    ax.set_xlabel('Sample Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Count', fontsize=14, fontweight='bold')
    ax.set_title('Preference Over Time (Cumulative)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'preference_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_dir / 'preference_timeline.png'}")


def generate_summary_stats(df, output_dir):
    """Generate summary statistics text file"""
    total_samples = len(df)
    choice_counts = df['actual_choice'].value_counts()

    baseline_count = choice_counts.get('baseline', 0)
    finetuned_count = choice_counts.get('finetuned', 0)
    equal_count = choice_counts.get('equal', 0)

    # Win rate (excluding equal)
    total_decisive = baseline_count + finetuned_count
    baseline_win_rate = (baseline_count / total_decisive * 100) if total_decisive > 0 else 0
    finetuned_win_rate = (finetuned_count / total_decisive * 100) if total_decisive > 0 else 0

    summary = f"""
Evaluation Summary
==================

Total Samples Evaluated: {total_samples}

Overall Preference:
-------------------
Baseline:  {baseline_count} ({baseline_count/total_samples*100:.1f}%)
Finetuned: {finetuned_count} ({finetuned_count/total_samples*100:.1f}%)
Equal:     {equal_count} ({equal_count/total_samples*100:.1f}%)

Win Rate (Excluding Equal):
---------------------------
Baseline:  {baseline_win_rate:.1f}%
Finetuned: {finetuned_win_rate:.1f}%

Decisive Choices: {total_decisive} / {total_samples} ({total_decisive/total_samples*100:.1f}%)
Equal Choices:    {equal_count} / {total_samples} ({equal_count/total_samples*100:.1f}%)

Conclusion:
-----------
"""

    if finetuned_count > baseline_count:
        winner = "Finetuned"
        margin = finetuned_count - baseline_count
    elif baseline_count > finetuned_count:
        winner = "Baseline"
        margin = baseline_count - finetuned_count
    else:
        winner = "Tie"
        margin = 0

    if winner != "Tie":
        summary += f"The {winner} model was preferred in {margin} more samples.\n"
    else:
        summary += "Both models received equal preference.\n"

    if total_decisive > 0:
        summary += f"When excluding equal choices, {winner} won {max(baseline_win_rate, finetuned_win_rate):.1f}% of the time.\n"

    summary_path = output_dir / 'summary_stats.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"\nSaved: {summary_path}")
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="Visualize human evaluation results")

    parser.add_argument("--csv", type=str, required=True, help="Path to evaluation CSV file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save visualizations (default: same as CSV)")

    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent / "visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading evaluation results from: {csv_path}")
    df = load_evaluation_results(csv_path)

    print(f"Found {len(df)} evaluation samples")
    print(f"\nGenerating visualizations in: {output_dir}\n")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'

    # Generate all plots
    plot_overall_preference(df, output_dir)
    plot_preference_percentage(df, output_dir)
    plot_win_rate(df, output_dir)
    plot_timeline(df, output_dir)
    generate_summary_stats(df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()


"""example usage

python src/f5_tts/eval/eval_visualize_results.py \
    --csv results/rap_evaluation/human_eval.csv \
    --output_dir results/rap_evaluation/visualizations

"""
