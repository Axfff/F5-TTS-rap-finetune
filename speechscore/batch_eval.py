"""Batch evaluation script for comparing baseline vs finetuned audio outputs."""

import argparse
import csv
import time
from pathlib import Path
import pprint

import pandas as pd
from tqdm import tqdm

from speechscore import SpeechScore


def load_audio_pairs(baseline_dir, finetuned_dir, ground_truth_dir=None):
    """Load pairs of baseline, finetuned, and optionally ground truth audio files"""
    baseline_path = Path(baseline_dir)
    finetuned_path = Path(finetuned_dir)

    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    if not finetuned_path.exists():
        raise FileNotFoundError(f"Finetuned directory not found: {finetuned_dir}")

    baseline_files = {f.stem: f for f in baseline_path.glob("*.wav")}
    finetuned_files = {f.stem: f for f in finetuned_path.glob("*.wav")}

    common_files = set(baseline_files.keys()) & set(finetuned_files.keys())

    if not common_files:
        raise ValueError("No matching audio files found between baseline and finetuned directories")

    audio_pairs = []
    for file_id in sorted(common_files):
        pair = {
            "id": file_id,
            "baseline": str(baseline_files[file_id]),
            "finetuned": str(finetuned_files[file_id]),
            "ground_truth": None,
        }

        if ground_truth_dir:
            gt_path = Path(ground_truth_dir)
            if gt_path.is_dir():
                gt_file = gt_path / "wavs" / f"{file_id}.wav"
                if not gt_file.exists():
                    gt_file = gt_path / f"{file_id}.wav"
                if gt_file.exists():
                    pair["ground_truth"] = str(gt_file)
            else:
                print(f"Warning: Ground truth directory not found: {ground_truth_dir}")

        audio_pairs.append(pair)

    return audio_pairs


def evaluate_audio_pair(pair, scorer, has_ground_truth):
    """Evaluate a single audio pair with specified metrics"""
    results = {"audio_id": pair["id"]}

    try:
        # Evaluate baseline
        if has_ground_truth and pair["ground_truth"]:
            baseline_score = scorer(
                test_path=pair["baseline"],
                reference_path=pair["ground_truth"],
                window=None,
                score_rate=16000
            )
        else:
            baseline_score = scorer(
                test_path=pair["baseline"],
                reference_path=None,
                window=None,
                score_rate=16000
            )

        # Evaluate finetuned
        if has_ground_truth and pair["ground_truth"]:
            finetuned_score = scorer(
                test_path=pair["finetuned"],
                reference_path=pair["ground_truth"],
                window=None,
                score_rate=16000
            )
        else:
            finetuned_score = scorer(
                test_path=pair["finetuned"],
                reference_path=None,
                window=None,
                score_rate=16000
            )

        # Process scores
        for metric_name in baseline_score.keys():
            if isinstance(baseline_score[metric_name], dict):
                for sub_key, sub_value in baseline_score[metric_name].items():
                    results[f"baseline_{metric_name}_{sub_key}"] = sub_value
                for sub_key, sub_value in finetuned_score[metric_name].items():
                    results[f"finetuned_{metric_name}_{sub_key}"] = sub_value
            else:
                results[f"baseline_{metric_name}"] = baseline_score[metric_name]
                results[f"finetuned_{metric_name}"] = finetuned_score[metric_name]

    except Exception as e:
        import traceback
        print(f"  Error evaluating {pair['id']}: {str(e)}")
        print(f"  Traceback: {traceback.format_exc()}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Objective metrics evaluation for baseline vs finetuned models")

    parser.add_argument("--baseline_dir", type=str, required=True, help="Directory containing baseline model outputs")
    parser.add_argument(
        "--finetuned_dir", type=str, required=True, help="Directory containing finetuned model outputs"
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Directory containing ground truth audio (required for intrusive metrics)",
    )
    parser.add_argument(
        "--output_csv", type=str, default="results/objective_metrics.csv", help="Path to save evaluation results CSV"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        choices=["all", "quality", "intelligibility", "timbre", "quick", "non_intrusive"],
        help="Metric category to evaluate",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("Objective Metrics Evaluation")
    print("=" * 80)

    # Define metric configurations
    metrics_config = {
        "quality": {
            "non_intrusive": ["DNSMOS", "SRMR"],
            "intrusive": ["PESQ", "FWSEGSNR", "SSNR", "SNR"],
        },
        "intelligibility": {
            "intrusive": ["STOI"],
        },
        "timbre": {
            "intrusive": ["MCD", "LSD"],
        },
        "signal": {
            "intrusive": ["SISDR"],
        },
    }

    # Select metrics based on user choice
    if args.metrics == "all":
        selected_metrics = []
        for category in metrics_config.values():
            selected_metrics.extend(category.get("non_intrusive", []))
            if args.ground_truth_dir:
                selected_metrics.extend(category.get("intrusive", []))
    elif args.metrics == "non_intrusive":
        selected_metrics = ["DNSMOS", "SRMR"]
    elif args.metrics == "quick":
        selected_metrics = ["DNSMOS", "SRMR"]
        if args.ground_truth_dir:
            selected_metrics.extend(["PESQ", "STOI", "MCD"])
    else:
        selected_metrics = []
        category_metrics = metrics_config.get(args.metrics, {})
        selected_metrics.extend(category_metrics.get("non_intrusive", []))
        if args.ground_truth_dir:
            selected_metrics.extend(category_metrics.get("intrusive", []))

    print(f"\nSelected metrics ({len(selected_metrics)}): {', '.join(selected_metrics)}")

    print(f"\nLoading audio pairs from:")
    print(f"  Baseline:      {args.baseline_dir}")
    print(f"  Finetuned:     {args.finetuned_dir}")
    if args.ground_truth_dir:
        print(f"  Ground Truth:  {args.ground_truth_dir}")
    else:
        print(f"  Ground Truth:  None (only non-intrusive metrics will be calculated)")

    audio_pairs = load_audio_pairs(args.baseline_dir, args.finetuned_dir, args.ground_truth_dir)
    print(f"\nFound {len(audio_pairs)} audio pairs")

    print("\nInitializing SpeechScore...")
    scorer = SpeechScore(selected_metrics)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating audio pairs...")
    print(f"Results will be saved to: {output_path}\n")

    all_results = []
    start_time = time.time()

    for pair in tqdm(audio_pairs, desc="Processing audio pairs"):
        print(f"\nProcessing: {pair['id']}")
        results = evaluate_audio_pair(pair, scorer, has_ground_truth=args.ground_truth_dir is not None)
        all_results.append(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Total audio pairs: {len(audio_pairs)}")
    print(f"Time elapsed: {elapsed_time / 60:.2f} minutes")
    print(f"Average time per pair: {elapsed_time / len(audio_pairs):.2f} seconds")
    print(f"\nResults saved to: {output_path}")

    print("\nSummary Statistics:")
    print("-" * 80)
    for col in df.columns:
        if col != "audio_id" and df[col].dtype in ["float64", "int64"]:
            baseline_col = col.startswith("baseline_")
            finetuned_col = col.startswith("finetuned_")

            if baseline_col or finetuned_col:
                mean_val = df[col].mean()
                std_val = df[col].std()
                print(f"{col:40s}: {mean_val:8.4f} ± {std_val:6.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()


"""Example usage:

# Non-intrusive metrics only (no ground truth needed)
python speechscore/batch_eval.py \
    --baseline_dir results/rap_evaluation/baseline \
    --finetuned_dir results/rap_evaluation/finetuned \
    --output_csv results/rap_evaluation/objective_metrics.csv \
    --metrics non_intrusive

# Quick evaluation (key metrics)
python speechscore/batch_eval.py \
    --baseline_dir results/rap_evaluation/baseline \
    --finetuned_dir results/rap_evaluation/finetuned \
    --ground_truth_dir data/test_rap_char \
    --output_csv results/rap_evaluation/objective_metrics.csv \
    --metrics quick

# Full evaluation (all metrics)
python speechscore/batch_eval.py \
    --baseline_dir results/rap_evaluation/baseline \
    --finetuned_dir results/rap_evaluation/finetuned \
    --ground_truth_dir data/test_rap_char \
    --output_csv results/rap_evaluation/objective_metrics.csv \
    --metrics all

# Only quality metrics
python speechscore/batch_eval.py \
    --baseline_dir results/rap_evaluation/baseline \
    --finetuned_dir results/rap_evaluation/finetuned \
    --ground_truth_dir data/test_rap_char \
    --output_csv results/rap_evaluation/objective_metrics.csv \
    --metrics quality
"""
