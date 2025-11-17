#!/usr/bin/env python3
"""
Strict sentence/word-aware audio slicing with Whisper (faster-whisper).

- Cuts at sentence punctuation when possible; otherwise at word boundaries.
- Hard-caps slice length within [min_sec, max_sec].
- Optional SRT export with per-cue limits.
- Fallback to energy-based slicing if Whisper misses too much speech.

Usage examples:
  python slice_strict.py input.wav --model medium --language en --min-sec 5 --max-sec 10 --mono --export-srt
  python slice_strict.py input.wav --vad --vad-threshold 0.25 --no-speech-threshold 0.0 --min-sec 6 --max-sec 12
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from faster_whisper import WhisperModel

PUNCT = set(".?!…，。！？；;：:")

# --------------------------- Whisper helpers ---------------------------

def words_from_segments(segments):
    """Flatten Whisper segments into [(start, end, text)] word list."""
    words = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                words.append((float(w.start), float(w.end), w.word))
    return words

# ------------------------ Slicing core (strict) ------------------------

def slice_words_strict(words: List[Tuple[float, float, str]],
                       min_s: float,
                       max_s: float) -> List[Tuple[float, float, str]]:
    """
    Greedy word-level slicing that NEVER exceeds max_s.
    Prefers to cut on punctuation after reaching min_s; otherwise cuts at last word <= max_s.
    Returns [(start, end, text)] in seconds.
    """
    out = []
    n = len(words)
    i = 0
    while i < n:
        st = words[i][0]
        last_punct = None
        j = i
        while j < n:
            end_j = words[j][1]
            dur = end_j - st

            # Track punctuation index
            wtxt = words[j][2]
            if wtxt and wtxt.strip() and wtxt.strip()[-1] in PUNCT:
                last_punct = j

            # If adding this word exceeds max, we must cut before j
            if dur > max_s:
                # Prefer punctuation cut inside window and >= min_s
                cut = last_punct
                if cut is None or (words[cut][1] - st) < min_s:
                    # fallback: last word that keeps <= max_s
                    cut = j - 1
                    # ensure we have at least one word
                    if cut < i:
                        cut = i
                    # (by construction, words[cut].end - st <= max_s now)
                end = words[cut][1]
                text = "".join(w[2] for w in words[i:cut + 1]).strip()
                out.append((st, end, text))
                i = cut + 1
                break

            j += 1

            # If we've reached end without exceeding max, we’ll flush later
        if j >= n:
            # We didn't exceed max; try to end the slice here.
            end = words[n - 1][1]
            dur = end - st
            if dur <= max_s:
                # If we already passed min and have a punctuation toward the end,
                # cut at punctuation; else cut at end.
                cut = None
                if dur >= min_s and last_punct is not None:
                    cut = last_punct
                if cut is None:
                    cut = n - 1
                end = words[cut][1]
                text = "".join(w[2] for w in words[i:cut + 1]).strip()
                out.append((st, end, text))
                i = cut + 1
            else:
                # dur > max_s (rare when last chunk is huge). Chunk it into <= max_s pieces.
                k = i
                while k < n:
                    cur_st = words[k][0]
                    # Push as far as we can within max_s
                    m = k
                    last_p = None
                    while m < n and (words[m][1] - cur_st) <= max_s:
                        if words[m][2].strip().endswith(tuple(PUNCT)):
                            last_p = m
                        m += 1
                    if last_p is not None and (words[last_p][1] - cur_st) >= min_s:
                        cut = last_p
                    else:
                        cut = max(k, m - 1)
                    cur_end = words[cut][1]
                    text = "".join(w[2] for w in words[k:cut + 1]).strip()
                    out.append((cur_st, cur_end, text))
                    k = cut + 1
                i = n
    return out

# ---------------------- Energy fallback (optional) ---------------------

def energy_fallback_slices(full: AudioSegment,
                           min_s: float,
                           max_s: float,
                           silence_db_from_avg: float = 16.0) -> List[Tuple[float, float, str]]:
    """
    If Whisper misses words, catch voiced regions by energy and split them into <= max_s windows.
    """
    thr = full.dBFS - silence_db_from_avg  # e.g., avg - 16 dB
    ns = detect_nonsilent(full, min_silence_len=150, silence_thresh=thr, seek_step=10)
    slices = []
    for s_ms, e_ms in ns:
        st, en = s_ms / 1000.0, e_ms / 1000.0
        cur = st
        while cur < en:
            nxt = min(en, cur + max_s)
            slices.append((cur, nxt, ""))  # text unknown here
            cur = nxt
    return slices

# ------------------------------ SRT I/O --------------------------------

def _wrap_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + (1 if cur else 0) + len(w) <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
            if len(lines) == max_lines:
                # soft truncate last line
                lines[-1] = (lines[-1] + " …")[:max_chars]
                return lines
    if cur:
        lines.append(cur)
    return lines[:max_lines]

def write_srt(cues: List[Tuple[float, float, str]],
              path: Path,
              max_chars: int,
              max_lines: int):
    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    lines = []
    for i, (st, en, tx) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt(st)} --> {fmt(en)}")
        wrapped = _wrap_lines(tx.strip() or "[no speech]", max_chars, max_lines)
        lines += wrapped
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

# ------------------------------ Main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Strict sentence/word-aware slicer with Whisper.")
    ap.add_argument("audio", help="Input audio (any ffmpeg-supported format)")
    ap.add_argument("--outdir", default="rap_slices", help="Output directory")

    # Whisper / inference
    ap.add_argument("--model", default="small", help="faster-whisper model (tiny/base/small/medium/large-v3 or path)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--compute-type", default=None, help="float16 (GPU), int8/int8_float16 (CPU)")
    ap.add_argument("--language", default=None, help="Force language code, e.g., en, zh")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--vad", action="store_true", help="Enable VAD pre-segmentation")
    ap.add_argument("--vad-threshold", type=float, default=0.30)
    ap.add_argument("--vad-min-silence-ms", type=int, default=150)
    ap.add_argument("--vad-speech-pad-ms", type=int, default=40)
    ap.add_argument("--no-speech-threshold", type=float, default=0.0, help="Lower keeps more low-conf speech")

    # Slicing constraints
    ap.add_argument("--min-sec", type=float, default=5.0)
    ap.add_argument("--max-sec", type=float, default=15.0)

    # Export options
    ap.add_argument("--pad-ms", type=int, default=80, help="Pad around cuts (ms)")
    ap.add_argument("--fade-ms", type=int, default=12, help="Fade in/out (ms)")
    ap.add_argument("--sr", type=int, default=16000, help="Output sample rate")
    ap.add_argument("--mono", action="store_true", help="Force mono")
    ap.add_argument("--export-srt", action="store_true", help="Write SRT alongside slices")
    ap.add_argument("--srt-max-chars", type=int, default=42)
    ap.add_argument("--srt-max-lines", type=int, default=2)

    args = ap.parse_args()

    in_path = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load audio for slicing/export
    full = AudioSegment.from_file(in_path)

    # Build Whisper model
    if args.device == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    else:
        dev = args.device
    compute_type = args.compute_type or ("float16" if dev == "cuda" else "int8")

    model = WhisperModel(args.model, device=dev, compute_type=compute_type)

    # Transcribe with words
    transcribe_kwargs = dict(
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=True,
        no_speech_threshold=args.no_speech_threshold,
    )
    if args.vad:
        transcribe_kwargs.update(dict(
            vad_filter=True,
            vad_parameters=dict(
                threshold=args.vad_threshold,
                min_silence_duration_ms=args.vad_min_silence_ms,
                speech_pad_ms=args.vad_speech_pad_ms,
            ),
        ))

    seg_gen, info = model.transcribe(str(in_path), **transcribe_kwargs)
    segments = list(seg_gen)
    words = words_from_segments(segments)

    # Build slices strictly within [min_sec, max_sec]
    slices: List[Tuple[float, float, str]] = []
    if words:
        slices = slice_words_strict(words, args.min_sec, args.max_sec)

    # Fallback if Whisper coverage is low
    whisper_speech = sum((s.end - s.start) for s in segments) if segments else 0.0
    covered = sum((en - st) for st, en, _ in slices) if slices else 0.0
    if (not slices) or (whisper_speech > 0 and covered < 0.5 * whisper_speech):
        print("↪️  Adding energy-based fallback slices (Whisper coverage seems low).")
        slices += energy_fallback_slices(full, args.min_sec, args.max_sec)

    # Sort & (light) de-dup by time
    slices.sort(key=lambda x: (x[0], x[1]))
    # Merge tiny overlaps
    merged = []
    for st, en, tx in slices:
        if merged and st <= merged[-1][1] + 1e-3:  # tiny overlap
            # extend end if needed, concatenate text
            prev_st, prev_en, prev_tx = merged[-1]
            if en > prev_en:
                merged[-1] = (prev_st, en, (prev_tx + " " + tx).strip())
        else:
            merged.append((st, en, tx))
    slices = merged

    # Final assert: enforce max-sec strictly (allow tiny float slack)
    max_over = max(((en - st) for st, en, _ in slices), default=0.0)
    if max_over > args.max_sec + 1e-3:
        print(f"⚠️ Warning: a slice exceeded max-sec ({max_over:.3f}s). This should not happen; please report.")

    # Export WAV slices + manifest
    manifest_path = outdir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "start_ms", "end_ms", "duration_ms", "text"])
        for i, (st, en, tx) in enumerate(slices, 1):
            pad = args.pad_ms
            start_ms = max(0, int(st * 1000) - pad)
            end_ms = min(len(full), int(en * 1000) + pad)
            seg = full[start_ms:end_ms].set_frame_rate(args.sr).set_sample_width(2)
            if args.mono:
                seg = seg.set_channels(1)
            if args.fade_ms > 0:
                seg = seg.fade_in(args.fade_ms).fade_out(args.fade_ms)
            out_name = f"{in_path.stem}_slice_{i:04d}_{start_ms}-{end_ms}.wav"
            seg.export(outdir / out_name, format="wav")
            w.writerow([out_name, start_ms, end_ms, end_ms - start_ms, tx])

    # Optional SRT (use the same slices, just wrap text lines)
    if args.export_srt:
        write_srt(slices, outdir / "slices.srt", args.srt_max_chars, args.srt_max_lines)

    print(f"✅ Exported {len(slices)} slices to {outdir}")
    print(f"🧾 Manifest: {manifest_path}")
    print(f"ℹ️ Whisper: lang={getattr(info,'language',None)}, device={dev}, compute={compute_type}; "
          f"segments={len(segments)}, words={len(words)}, speech≈{whisper_speech:.1f}s, sliced≈{covered:.1f}s")

if __name__ == "__main__":
    main()