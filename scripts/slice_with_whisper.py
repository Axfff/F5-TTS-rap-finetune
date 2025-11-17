#!/usr/bin/env python3
import argparse, csv, os, re
from pathlib import Path
from typing import List, Tuple
from pydub import AudioSegment
from faster_whisper import WhisperModel

PUNCT = set(list(".?!…，。！？；;：:"))  # prefer splitting after these

def merge_to_target_windows(segments, min_s, max_s):
    """
    Greedy merge whisper segments into windows between [min_s, max_s],
    preferring to cut when the cumulative text ends with punctuation.
    Returns list of (start, end, text).
    """
    out = []
    cur_start, cur_end, cur_text = None, None, []
    def flush():
        nonlocal cur_start, cur_end, cur_text
        if cur_start is not None:
            text = re.sub(r"\s+", " ", " ".join(cur_text)).strip()
            out.append((cur_start, cur_end, text))
        cur_start, cur_end, cur_text = None, None, []

    for s in segments:
        st, en, tx = float(s.start), float(s.end), s.text or ""
        if cur_start is None:
            cur_start = st
        cur_end = en
        cur_text.append(tx.strip())
        dur = cur_end - cur_start
        ends_with_punct = (cur_text and len(cur_text[-1])>0 and cur_text[-1][-1] in PUNCT)

        if dur >= min_s and (ends_with_punct or dur >= max_s):
            flush()

    # leftover
    if cur_start is not None:
        # try to merge with previous if short
        if out and (cur_end - out[-1][0]) <= max_s:
            prev = out.pop()
            merged = (prev[0], cur_end, (prev[2] + " " + re.sub(r"\s+"," "," ".join(cur_text))).strip())
            out.append(merged)
        else:
            flush()
    return out

def write_srt(slices: List[Tuple[float,float,str]], path: Path):
    def fmt(t):
        h=int(t//3600); m=int((t%3600)//60); s=t%60
        return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")
    lines=[]
    for i,(st,en,tx) in enumerate(slices,1):
        lines.append(str(i))
        lines.append(f"{fmt(st)} --> {fmt(en)}")
        lines.append(tx.strip() or "[no speech]")
        lines.append("")  # blank line
    path.write_text("\n".join(lines), encoding="utf-8")

def words_from_segments(segments):
    """Flatten whisper segments into a list of (start, end, text) words with times."""
    words = []
    for seg in segments:
        if getattr(seg, "words", None):
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                # whisper words often include leading spaces; keep them for readable text
                words.append((float(w.start), float(w.end), w.word))
    return words

def merge_words_to_windows(words, min_s, max_s):
    """
    Greedy pack words into windows [min_s, max_s].
    Prefer to cut on punctuation; otherwise hard-cut at max_s.
    Returns [(start, end, text), ...] with 'end-start' <= max_s (up to tiny rounding).
    """
    out = []
    i, n = 0, len(words)
    while i < n:
        st = words[i][0]
        j = i
        last_punct_idx = None
        while j < n:
            end = words[j][1]
            dur = end - st
            wtxt = words[j][2]
            if wtxt and wtxt.strip() and wtxt.strip()[-1] in PUNCT:
                last_punct_idx = j

            if dur >= min_s and (last_punct_idx is not None or dur >= max_s):
                # choose punctuation if available and within max_s, else hard-cut at j
                cut = last_punct_idx
                if cut is None or (words[cut][1] - st) > max_s:
                    # hard-cap at max_s by backing up to the last word <= max_s
                    cut = j
                    while cut > i and (words[cut][1] - st) > max_s:
                        cut -= 1
                end = words[cut][1]
                text = "".join(w[2] for w in words[i:cut+1]).strip()
                out.append((st, end, text))
                i = cut + 1
                break

            j += 1

        if j >= n and i < n:
            # flush remainder; also enforce max_s by chunking if needed
            cur_st = st
            k = i
            while k < n:
                # find the furthest word that keeps <= max_s
                m = k
                while m < n and (words[m][1] - cur_st) <= max_s:
                    m += 1
                if m == k:  # single word longer than cap (extremely rare) -> force single-word slice
                    m = k + 1
                end = words[m-1][1]
                text = "".join(w[2] for w in words[k:m]).strip()
                out.append((cur_st, end, text))
                k = m
                if k < n:
                    cur_st = words[k][0]
            i = n
    return out

def main():
    ap = argparse.ArgumentParser(description="Slice long rap audio with Whisper-aligned sentence boundaries.")
    ap.add_argument("audio", help="Input audio file (any ffmpeg-supported)")
    ap.add_argument("--outdir", default="rap_slices", help="Output directory")
    ap.add_argument("--model", default="small", help="faster-whisper model (tiny/base/small/medium/large-v3 or local path)")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Prefer cuda if available")
    ap.add_argument("--compute-type", default=None, help="e.g. float16 (GPU), int8/int8_float16 (CPU)")
    ap.add_argument("--language", default=None, help="Force language code (e.g. en, zh). Default: auto-detect")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--vad", action="store_true", help="Enable VAD filter (helps with dense mixes)")
    ap.add_argument("--min-sec", type=float, default=5.0)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--pad-ms", type=int, default=80, help="Add small margin around cuts (ms)")
    ap.add_argument("--sr", type=int, default=16000, help="Output sample rate")
    ap.add_argument("--mono", action="store_true", help="Force mono output (good for speech/TTS)")
    ap.add_argument("--export-srt", action="store_true", help="Also write an SRT of final slices")
    args = ap.parse_args()

    in_path = Path(args.audio)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load audio with pydub for slicing/export
    full = AudioSegment.from_file(in_path)

    # Build whisper model
    device = ("cuda" if args.device=="auto" else args.device)
    if args.device=="auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    compute_type = args.compute_type or ("float16" if device=="cuda" else "int8")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    # Transcribe
    seg_gen, info = model.transcribe(
        str(in_path),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=args.vad,
        word_timestamps=True,
    )
    segments = list(seg_gen)
    if not segments:
        print("No speech segments found.")
        return

    # Merge segments into 5–15 s windows with punctuation preference
    words = words_from_segments(segments)
    if not words:
        # fall back to your old segment-level merge if no word times
        slices = merge_to_target_windows(segments, args.min_sec, args.max_sec)
    else:
        slices = merge_words_to_windows(words, args.min_sec, args.max_sec)

    # Export audio slices + manifest
    manifest_path = outdir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename","start_ms","end_ms","duration_ms","text"])
        for i, (st, en, tx) in enumerate(slices, 1):
            # Add small padding but clamp to file
            pad = args.pad_ms
            start_ms = max(0, int(st*1000) - pad)
            end_ms   = min(len(full), int(en*1000) + pad)
            seg = full[start_ms:end_ms]
            seg = seg.set_frame_rate(args.sr).set_sample_width(2)  # 16-bit PCM
            if args.mono: seg = seg.set_channels(1)
            seg = seg.fade_in(12).fade_out(12)

            out_name = f"{in_path.stem}_slice_{i:04d}_{start_ms}-{end_ms}.wav"
            seg.export(outdir / out_name, format="wav")
            w.writerow([out_name, start_ms, end_ms, end_ms-start_ms, tx])

    if args.export_srt:
        write_srt(slices, outdir / "slices.srt")

    print(f"✅ Exported {len(slices)} slices to {outdir}")
    print(f"🧾 Manifest: {manifest_path}")
    if args.export_srt:
        print(f"📝 SRT: {outdir/'slices.srt'}")
    print(f"ℹ️ Language: {info.language}, prob={getattr(info,'language_probability',None)}; device={device}, compute_type={compute_type}")

if __name__ == "__main__":
    main()