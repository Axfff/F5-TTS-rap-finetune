#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------- helpers ---------------------------

def which(cmd: str) -> Optional[str]:
    from shutil import which as _which
    return _which(cmd)

def list_wavs(inp: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.wav" if recursive else "*.wav"
    return [Path(p) for p in glob(str(inp / pattern), recursive=recursive)]

def run(cmd: List[str]) -> Tuple[int, str, str]:
    env = os.environ.copy()
    env["TORCHAUDIO_USE_SOUNDFILE"] = "1"
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def expected_demucs_root(out_dir: Path, model: str) -> Path:
    # demucs writes under: out_dir/<model>/
    return out_dir / model

def collect_vocals(root: Path) -> List[Path]:
    """Find all vocals.wav under a directory tree."""
    return [Path(p) for p in glob(str(root / "**/vocals.wav"), recursive=True)]

def copy_flat(vocals_paths: List[Path], flat_dir: Path, out_format: str = "{idx:04d}_{name}.wav") -> List[Tuple[Path, Path]]:
    flat_dir.mkdir(parents=True, exist_ok=True)
    mapping = []
    for idx, vp in enumerate(sorted(vocals_paths), 1):
        # Build a stable file name based on original parent folder name + base
        # Typically vp = .../<track>/vocals.wav ; use grandparent+track to help uniqueness
        parent = vp.parent.name
        gparent = vp.parent.parent.name if vp.parent.parent else ""
        src_name = f"{gparent}__{parent}__vocals".strip("_")
        out_name = out_format.format(idx=idx, name=src_name).replace(os.sep, "_")
        dst = flat_dir / out_name
        # Avoid overwriting different sources with same name
        if dst.exists():
            # Append a counter
            k = 2
            stem = dst.stem
            while (flat_dir / f"{stem}__{k}.wav").exists():
                k += 1
            dst = flat_dir / f"{stem}__{k}.wav"
        shutil.copy2(vp, dst)
        mapping.append((vp, dst))
    return mapping

# --------------------------- backends ---------------------------

def extract_with_demucs(wav: Path, out_dir: Path, model: str, device: str, jobs: int, force: bool) -> Tuple[bool, str]:
    """
    Returns (success, message). Demucs command example:
    demucs --two-stems=vocals -n htdemucs -o out_dir --device cpu/0 FILE.wav
    """
    if not which("demucs"):
        return False, "demucs not found. Install via: pip install demucs"

    # Skip if we detect vocals already extracted for this track (best-effort)
    model_root = expected_demucs_root(out_dir, model)
    existing = list((model_root).glob(f"**/{wav.stem}/vocals.wav"))
    if existing and not force:
        return True, f"skip (exists): {wav}"

    cmd = [
        "demucs",
        "--two-stems=vocals",
        "-n", model,
        "-o", str(out_dir),
    ]
    # device: "cpu" or "cuda" or "0" (GPU index)
    if device:
        cmd += ["--device", device]
    if jobs and jobs > 0:
        cmd += ["-j", str(jobs)]
    cmd.append(str(wav))

    rc, out, err = run(cmd)
    if rc != 0:
        return False, f"demucs failed for {wav}:\n{err or out}"
    return True, f"ok: {wav}"

def extract_with_spleeter(wav: Path, out_dir: Path, force: bool) -> Tuple[bool, str]:
    """
    Returns (success, message). Spleeter command example:
    spleeter separate -p spleeter:2stems -o out_dir FILE.wav
    Output: out_dir/FILE/vocals.wav
    """
    if not which("spleeter"):
        return False, "spleeter not found. Install via: pip install spleeter"

    # Skip if exists
    target = out_dir / wav.stem / "vocals.wav"
    if target.exists() and not force:
        return True, f"skip (exists): {wav}"

    cmd = [
        "spleeter", "separate",
        "-p", "spleeter:2stems",
        "-o", str(out_dir),
        str(wav),
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        return False, f"spleeter failed for {wav}:\n{err or out}"
    return True, f"ok: {wav}"

# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch extract vocals for all WAVs in a folder.")
    ap.add_argument("input_dir", help="Folder containing .wav files (processed recursively by default)")
    ap.add_argument("-o", "--outdir", default="stems_out", help="Output root directory")
    ap.add_argument("--backend", choices=["demucs", "spleeter"], default="demucs", help="Separation backend")
    ap.add_argument("--model", default="htdemucs", help="Demucs model name (e.g., htdemucs, htdemucs_ft, mdx_extra)")
    ap.add_argument("--device", default="", help='Demucs device: "cpu", "cuda", or GPU index (e.g., "0")')
    ap.add_argument("-j", "--jobs", type=int, default=0, help="Demucs internal jobs (threads). 0=auto")
    ap.add_argument("-p", "--parallel", type=int, default=2, help="How many files to process in parallel")
    ap.add_argument("--no-recursive", action="store_true", help="Do not recurse into subfolders")
    ap.add_argument("--flat", action="store_true", help="Copy all vocals into a flat folder (_vocals_flat)")
    ap.add_argument("--force", action="store_true", help="Re-run even if an output exists")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.outdir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = list_wavs(in_dir, recursive=(not args.no_recursive))
    if not wavs:
        print(f"No .wav files found under: {in_dir}")
        sys.exit(1)

    print(f"Found {len(wavs)} wav(s). Backend={args.backend}")

    # Process in parallel (by file). Each process still allows Demucs to use internal -j threads.
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        futs = []
        for w in wavs:
            if args.backend == "demucs":
                fut = ex.submit(extract_with_demucs, w, out_dir, args.model, args.device, args.jobs, args.force)
            else:
                fut = ex.submit(extract_with_spleeter, w, out_dir, args.force)
            futs.append((w, fut))

        for w, fut in futs:
            ok, msg = fut.result()
            results.append((w, ok, msg))
            print(("✅" if ok else "❌"), msg)

    # Collect vocals and produce manifest
    vocals_found = []
    if args.backend == "demucs":
        vlist = collect_vocals(expected_demucs_root(out_dir, args.model))
        vocals_found.extend(vlist)
    else:
        vocals_found.extend(collect_vocals(out_dir))

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["input_wav", "vocals_wav"])
        # Best-effort mapping: match by name when possible
        # Build index by stem for fast pairing
        idx = {vp.parent.name: vp for vp in vocals_found}  # key ~ original basename
        for src, ok, _msg in results:
            vp = None
            # demucs/spleeter convention puts vocals at .../<src.stem>/vocals.wav
            if src.stem in idx:
                vp = idx[src.stem]
            else:
                # fallback: find any vocals that contain stem in path
                for cand in vocals_found:
                    if src.stem in str(cand):
                        vp = cand
                        break
            w.writerow([str(src), str(vp) if vp else ""])

    # Optional flat copies
    if args.flat:
        flat_dir = out_dir / "_vocals_flat"
        copies = copy_flat(vocals_found, flat_dir)
        print(f"Copied {len(copies)} vocals to: {flat_dir}")

    print(f"Done. Manifest: {manifest}")
    if args.backend == "demucs":
        print(f"Demucs outputs under: {expected_demucs_root(out_dir, args.model)}")
    else:
        print(f"Spleeter outputs under: {out_dir}")

if __name__ == "__main__":
    main()
