#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import List

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def detect_silence_boundaries(
    audio: AudioSegment,
    min_silence_len_ms: int,
    silence_thresh_db: float,
):
    """
    Return 'nonsilent' intervals and a set of candidate cut points
    (starts/ends of nonsilent chunks), which tend to be near silence.
    """
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len_ms,
        silence_thresh=silence_thresh_db,
        seek_step=5,  # ms; smaller = finer but slower
    )
    # Flatten to a set of candidate cut points (ms)
    cut_points = set()
    for start, end in nonsilent:
        cut_points.add(start)
        cut_points.add(end)
    # Always allow cutting at file boundaries
    cut_points.add(0)
    cut_points.add(len(audio))
    return sorted(nonsilent), sorted(cut_points)


def choose_segments_by_window(
    audio_len_ms: int,
    cut_points: List[int],
    min_len_ms: int,
    max_len_ms: int,
):
    """
    Greedy packing: move a window [cur+min, cur+max], prefer the *latest*
    allowed cut point inside the window; if none, force cut at cur+max (or EOF).
    """
    segments = []
    cur = 0
    cps = cut_points

    while cur < audio_len_ms:
        lower = cur + min_len_ms
        upper = min(cur + max_len_ms, audio_len_ms)

        # all candidates within the window
        candidates = [p for p in cps if lower <= p <= upper]
        if candidates:
            nxt = max(candidates)  # prefer the farthest (largest slice <= max)
        else:
            nxt = upper  # forced cut

        if nxt <= cur:  # safety
            nxt = min(cur + max_len_ms, audio_len_ms)

        segments.append((cur, nxt))
        cur = nxt

    return segments


def trim_edges_keep_silence(seg: AudioSegment, keep_ms: int, fade_ms: int):
    """
    Optional: add tiny keep_silence and fade in/out to avoid clicks.
    """
    if keep_ms > 0:
        # pad tiny silence on both sides
        silence = AudioSegment.silent(duration=keep_ms, frame_rate=seg.frame_rate)
        seg = silence + seg + silence
    if fade_ms > 0:
        seg = seg.fade_in(fade_ms).fade_out(fade_ms)
    return seg


def main():
    ap = argparse.ArgumentParser(
        description="Split a rap audio into 5–15s chunks (silence-aware)."
    )
    ap.add_argument("input", type=str, help="Path to input audio (any ffmpeg format)")
    ap.add_argument("-o", "--outdir", type=str, default="slices", help="Output folder")
    ap.add_argument("--min-sec", type=float, default=5.0, help="Min slice length (s)")
    ap.add_argument("--max-sec", type=float, default=15.0, help="Max slice length (s)")
    ap.add_argument(
        "--min-silence-len",
        type=int,
        default=150,
        help="Minimum silence to detect (ms). Rap often needs small values (100–250).",
    )
    ap.add_argument(
        "--silence-thresh",
        type=float,
        default=None,
        help=(
            "Silence threshold in dBFS (e.g., -35). "
            "Default: computed as (audio.dBFS - 14)."
        ),
    )
    ap.add_argument(
        "--keep-silence",
        type=int,
        default=120,
        help="Keep small silence (ms) at cuts to avoid pops.",
    )
    ap.add_argument(
        "--fade-ms",
        type=int,
        default=15,
        help="Fade in/out at segment edges (ms) to avoid clicks.",
    )
    ap.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Output sample rate (Hz), e.g. 16000/22050/44100.",
    )
    ap.add_argument(
        "--mono",
        action="store_true",
        help="Force mono (recommended for speech/TTS finetuning).",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="manifest.csv",
        help="CSV manifest file to write (relative to outdir).",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    audio = AudioSegment.from_file(in_path)
    # Compute default silence threshold if not provided
    # A dynamic rule: threshold = avg dBFS - 14 (tweak if your mix is loud/quiet)
    if args.silence_thresh is None:
        silence_thresh_db = audio.dBFS - 14.0
    else:
        silence_thresh_db = args.silence_thresh

    min_len_ms = int(args.min_sec * 1000)
    max_len_ms = int(args.max_sec * 1000)

    # Detect nonsilent + build candidate cut points
    nonsilent, cut_points = detect_silence_boundaries(
        audio, args.min_silence_len, silence_thresh_db
    )

    # Build segments by windowing with preferred cut points
    segments = choose_segments_by_window(len(audio), cut_points, min_len_ms, max_len_ms)

    # Export segments + manifest
    base = in_path.stem
    manifest_path = out_dir / args.manifest
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "start_ms", "end_ms", "duration_ms"])
        for i, (start, end) in enumerate(segments, 1):
            seg = audio[start:end]
            seg = seg.set_frame_rate(args.sr).set_sample_width(2)  # 16-bit PCM
            if args.mono:
                seg = seg.set_channels(1)
            # edge polish
            seg = trim_edges_keep_silence(seg, keep_ms=args.keep_silence, fade_ms=args.fade_ms)

            out_name = f"{base}_slice_{i:04d}_{start}-{end}.wav"
            out_path = out_dir / out_name
            seg.export(out_path, format="wav")
            w.writerow([out_name, start, end, end - start])

    print(f"✅ Done. Wrote {len(segments)} slices to: {out_dir}")
    print(f"🧾 Manifest: {manifest_path}")
    print(
        f"🔧 Tips: If slices are too long/short, tweak --min-silence-len and --silence-thresh "
        f"(current: {silence_thresh_db:.1f} dBFS)."
    )


if __name__ == "__main__":
    main()