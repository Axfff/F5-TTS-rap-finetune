import os
import sys

sys.path.append(os.getcwd())

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer, convert_char_to_pinyin


target_rms = 0.1
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024


def load_model_from_config(
    config_name,
    ckpt_path,
    vocab_file,
    mel_spec_type="vocos",
    ode_method="euler",
    use_ema=True,
    device="cuda",
):
    """Load model from config and checkpoint"""
    from importlib.resources import files
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{config_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    tokenizer = "custom" if vocab_file else model_cfg.model.tokenizer
    dataset_name = vocab_file if vocab_file else model_cfg.datasets.name

    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)

    return model


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


def preprocess_audio(audio_path, device):
    """Load and preprocess reference audio"""
    audio, sr = torchaudio.load(audio_path)

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms

    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    audio = audio.to(device)

    return audio, rms


def inference_single(
    model,
    vocoder,
    ref_audio,
    ref_text,
    gen_text,
    ref_rms,
    nfe_step=32,
    cfg_strength=2.0,
    sway_sampling_coef=-1.0,
    speed=1.0,
    mel_spec_type="vocos",
    device="cuda",
):
    """Run inference for a single sample"""
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "

    text_list = [ref_text + gen_text]
    final_text_list = convert_char_to_pinyin(text_list)

    ref_audio_len = ref_audio.shape[-1] // hop_length
    ref_text_len = len(ref_text.encode("utf-8"))
    gen_text_len = len(gen_text.encode("utf-8"))
    duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=ref_audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )

        generated = generated.to(torch.float32)
        generated = generated[:, ref_audio_len:, :]
        generated_mel = generated.permute(0, 2, 1)

        if mel_spec_type == "vocos":
            generated_wave = vocoder.decode(generated_mel)
        elif mel_spec_type == "bigvgan":
            generated_wave = vocoder(generated_mel)

        if ref_rms < target_rms:
            generated_wave = generated_wave * ref_rms / target_rms

        generated_wave = generated_wave.squeeze().cpu()

    return generated_wave


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
    parser.add_argument("--ode_method", type=str, default="euler", help="ODE solver method")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights")

    parser.add_argument("--ref_text_ratio", type=float, default=0.5, help="Ratio of text to use as reference (0-1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--local_vocoder", action="store_true", help="Use local vocoder checkpoint")
    parser.add_argument("--vocoder_path", type=str, default="", help="Local vocoder path")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n" + "="*80)
    print("Loading evaluation dataset...")
    print("="*80)
    eval_samples = load_eval_dataset(args.eval_data_dir)
    print(f"Loaded {len(eval_samples)} evaluation samples")

    print("\n" + "="*80)
    print("Loading vocoder...")
    print("="*80)
    if args.mel_spec_type == "vocos":
        vocoder_local_path = args.vocoder_path or "../checkpoints/charactr/vocos-mel-24khz"
    elif args.mel_spec_type == "bigvgan":
        vocoder_local_path = args.vocoder_path or "../checkpoints/bigvgan_v2_24khz_100band_256x"

    vocoder = load_vocoder(
        vocoder_name=args.mel_spec_type,
        is_local=args.local_vocoder,
        local_path=vocoder_local_path,
        device=device
    )

    print("\n" + "="*80)
    print("Loading baseline model...")
    print("="*80)
    print(f"Checkpoint: {args.baseline_ckpt}")
    baseline_model = load_model_from_config(
        config_name=args.config_name,
        ckpt_path=args.baseline_ckpt,
        vocab_file=args.vocab_file,
        mel_spec_type=args.mel_spec_type,
        ode_method=args.ode_method,
        use_ema=args.use_ema,
        device=device,
    )

    print("\n" + "="*80)
    print("Loading finetuned model...")
    print("="*80)
    print(f"Checkpoint: {args.finetuned_ckpt}")
    finetuned_model = load_model_from_config(
        config_name=args.config_name,
        ckpt_path=args.finetuned_ckpt,
        vocab_file=args.vocab_file,
        mel_spec_type=args.mel_spec_type,
        ode_method=args.ode_method,
        use_ema=args.use_ema,
        device=device,
    )

    baseline_output_dir = Path(args.output_dir) / "baseline"
    finetuned_output_dir = Path(args.output_dir) / "finetuned"
    baseline_output_dir.mkdir(parents=True, exist_ok=True)
    finetuned_output_dir.mkdir(parents=True, exist_ok=True)

    baseline_model.eval()
    finetuned_model.eval()

    print("\n" + "="*80)
    print("Starting batch inference...")
    print("="*80)
    print(f"Baseline output: {baseline_output_dir}")
    print(f"Finetuned output: {finetuned_output_dir}")
    print(f"Reference text ratio: {args.ref_text_ratio}")
    print()

    start_time = time.time()

    for sample in tqdm(eval_samples, desc="Processing samples"):
        audio_file = sample["audio_file"]
        audio_path = sample["audio_path"]
        full_text = sample["text"]

        ref_audio, ref_rms = preprocess_audio(audio_path, device)

        split_idx = int(len(full_text) * args.ref_text_ratio)
        if split_idx == 0:
            split_idx = min(10, len(full_text) // 2)

        ref_text = full_text[:split_idx]
        gen_text = full_text[split_idx:]

        if not gen_text.strip():
            print(f"Skipping {audio_file}: no generation text after split")
            continue

        baseline_wave = inference_single(
            model=baseline_model,
            vocoder=vocoder,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            ref_rms=ref_rms,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
            mel_spec_type=args.mel_spec_type,
            device=device,
        )

        finetuned_wave = inference_single(
            model=finetuned_model,
            vocoder=vocoder,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            ref_rms=ref_rms,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
            mel_spec_type=args.mel_spec_type,
            device=device,
        )

        torchaudio.save(
            baseline_output_dir / f"{audio_file}.wav",
            baseline_wave.unsqueeze(0),
            target_sample_rate
        )

        torchaudio.save(
            finetuned_output_dir / f"{audio_file}.wav",
            finetuned_wave.unsqueeze(0),
            target_sample_rate
        )

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
    
"""
