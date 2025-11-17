import os
import sys

sys.path.append(os.getcwd())

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def run_inference_cli(ckpt_path, vocab_file, ref_audio, ref_text, gen_text, output_path,
                       mel_spec_type="vocos", nfe_step=32, cfg_strength=2.0,
                       sway_sampling_coef=-1.0, speed=1.0):
    """Run inference using the CLI command"""
    cmd = [
        "f5-tts_infer-cli",
        "-p", ckpt_path,
        "-r", ref_audio,
        "-s", ref_text,
        "-t", gen_text,
        "-o", str(Path(output_path).parent),
        "-w", Path(output_path).name,
        "--vocoder_name", mel_spec_type,
        "--nfe_step", str(nfe_step),
        "--cfg_strength", str(cfg_strength),
        "--sway_sampling_coef", str(sway_sampling_coef),
        "--speed", str(speed),
    ]

    if vocab_file:
        cmd.extend(["-v", vocab_file])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running CLI: {result.stderr}")
        return False

    return True


def load_eval_dataset(data_dir):
    """Load evaluation dataset from metadata.csv"""
    metadata_path = Path(data_dir) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_dir}")

    df = pd.read_csv(metadata_path, sep="|", header=None, names=["audio_file", "text"])

    eval_samples = []
    for _, row in df.iterrows():
        audio_path = Path(data_dir) / "wavs" / f"{row['audio_file']}.wav"
        if audio_path.exists():
            eval_samples.append({
                "audio_file": row["audio_file"],
                "audio_path": str(audio_path),
                "text": row["text"]
            })

    return eval_samples


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and finetuned model inference")

    parser.add_argument("--eval_data_dir", type=str, required=True, help="Path to evaluation dataset directory")
    parser.add_argument("--baseline_ckpt", type=str, required=True, help="Path to baseline pretrained checkpoint")
    parser.add_argument("--finetuned_ckpt", type=str, required=True, help="Path to finetuned checkpoint")
    parser.add_argument("--config_name", type=str, default="F5TTS_Base", help="Model config name")
    parser.add_argument("--vocab_file", type=str, default="", help="Custom vocab file path (leave empty to use default)")
    parser.add_argument("--output_dir", type=str, default="results/comparison", help="Output directory for generated audio")

    parser.add_argument("--mel_spec_type", type=str, default="vocos", choices=["vocos", "bigvgan"])
    parser.add_argument("--nfe_step", type=int, default=32, help="Number of function evaluations")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="Classifier-free guidance strength")
    parser.add_argument("--sway_sampling_coef", type=float, default=-1.0, help="Sway sampling coefficient")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Loading evaluation dataset...")
    print("="*80)
    eval_samples = load_eval_dataset(args.eval_data_dir)
    print(f"Loaded {len(eval_samples)} evaluation samples")

    baseline_output_dir = Path(args.output_dir) / "baseline"
    finetuned_output_dir = Path(args.output_dir) / "finetuned"
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    finetuned_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("Starting batch inference using CLI...")
    print("="*80)
    print(f"Baseline checkpoint: {args.baseline_ckpt}")
    print(f"Finetuned checkpoint: {args.finetuned_ckpt}")
    print(f"Baseline output: {baseline_output_dir}")
    print(f"Finetuned output: {finetuned_output_dir}")
    print()

    start_time = time.time()

    for sample in tqdm(eval_samples, desc="Processing samples"):
        audio_file = sample["audio_file"]
        audio_path = sample["audio_path"]
        full_text = sample["text"]

        # Simple approach: use full audio and text as reference, regenerate the same text
        ref_text = full_text
        gen_text = full_text

        # Run baseline inference via CLI
        baseline_output = baseline_output_dir / f"{audio_file}.wav"
        success = run_inference_cli(
            ckpt_path=args.baseline_ckpt,
            vocab_file=args.vocab_file,
            ref_audio=audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            output_path=baseline_output,
            mel_spec_type=args.mel_spec_type,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
        )

        if not success:
            print(f"Failed to generate baseline audio for {audio_file}")
            continue

        # Run finetuned inference via CLI
        finetuned_output = finetuned_output_dir / f"{audio_file}.wav"
        success = run_inference_cli(
            ckpt_path=args.finetuned_ckpt,
            vocab_file=args.vocab_file,
            ref_audio=audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            output_path=finetuned_output,
            mel_spec_type=args.mel_spec_type,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
        )

        if not success:
            print(f"Failed to generate finetuned audio for {audio_file}")

    elapsed_time = time.time() - start_time

    print("\n" + "="*80)
    print("Batch inference completed!")
    print("="*80)
    print(f"Total samples: {len(eval_samples)}")
    print(f"Time elapsed: {elapsed_time / 60:.2f} minutes")
    print(f"Average time per sample: {elapsed_time / len(eval_samples):.2f} seconds")
    print(f"\nBaseline outputs: {baseline_output_dir}")
    print(f"Finetuned outputs: {finetuned_output_dir}")


if __name__ == "__main__":
    main()

"""example usage

python src/f5_tts/eval/eval_compare_baseline_finetuned.py \
    --eval_data_dir data/test_rap_char \
    --baseline_ckpt ckpts/F5TTS_Base/model_1200000.pt \
    --finetuned_ckpt path/to/your/finetuned.ckpt \
    --vocab_file data/rap_1107_char/vocab.txt \
    --output_dir results/rap_evaluation \
    --ref_text_ratio 0.5

python src/f5_tts/eval/eval_compare_baseline_finetuned.py \
    --eval_data_dir data/clean_rap_1111_char/ \
    --baseline_ckpt ckpts/clean_rap_1111/pretrained_model_1250000.safetensors \
    --finetuned_ckpt ckpts/clean_rap_1111/model_last.pt \
    --vocab_file data/clean_rap_1111_char/vocab.txt \
    --output_dir results/rap_evaluation \
    --ref_text_ratio 0.5
    
"""
