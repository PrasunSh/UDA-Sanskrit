#!/usr/bin/env python
"""
Convert the KB .m4a corpus to the format the paper uses:
    mono, 16-bit PCM WAV, 8 kHz  (Sec 3.1: "downsampled to 8 kHz ... in all our experiments")

Input layout (as found on disk):
    <src_root>/<lang>/<split>/audio/<utt_id>.m4a        e.g. .../hindi/train/audio/8444...-590-f.m4a
Output layout:
    <dst_root>/<lang>/<split>/<utt_id>.wav
Also writes a manifest per lang/split:
    <dst_root>/<lang>/<split>/manifest.csv   with columns: utt_id,gender,wav_path

The utt_id (filename stem) is preserved verbatim, so the trailing -m / -f gender tag
is retained and also parsed into a `gender` column.

Usage examples (run from anywhere):
    python convert_audio.py --langs sanskrit --splits train
    python convert_audio.py --langs hindi --splits train --limit 15000
    python convert_audio.py --langs hindi sanskrit telugu --workers 8
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_SRC = r"D:\train_audio\kb_data_clean_m4a"
DEFAULT_DST = r"D:\train_wav_8k"
TARGET_SR = 8000

# Known winget install location, used as a fallback when ffmpeg isn't on PATH.
_WINGET_FFMPEG = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    r"Microsoft\WinGet\Packages",
    r"Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe",
    r"ffmpeg-9.0-full_build\bin\ffmpeg.exe",
)


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    if os.path.isfile(_WINGET_FFMPEG):
        return _WINGET_FFMPEG
    # last resort: search winget packages dir for any ffmpeg.exe
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Microsoft\WinGet\Packages")
    for root, _, files in os.walk(base):
        if "ffmpeg.exe" in files:
            return os.path.join(root, "ffmpeg.exe")
    raise FileNotFoundError("ffmpeg not found on PATH or in winget packages.")


def gender_of(utt_id):
    tail = utt_id.rsplit("-", 1)[-1].lower()
    return {"m": "male", "f": "female"}.get(tail, "unknown")


def convert_one(args):
    ffmpeg, src, dst, sr = args
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return ("skip", src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src,
           "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", dst]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0:
        return ("fail", f"{src} :: {r.stderr.decode(errors='ignore').strip()[:200]}")
    return ("ok", src)


def gather_jobs(ffmpeg, src_root, dst_root, langs, splits, sr, limit):
    jobs, manifests = [], {}
    for lang in langs:
        for split in splits:
            audio_dir = os.path.join(src_root, lang, split, "audio")
            if not os.path.isdir(audio_dir):
                print(f"  (skip) no audio dir: {audio_dir}")
                continue
            files = sorted(f for f in os.listdir(audio_dir) if f.lower().endswith(".m4a"))
            if limit:
                files = files[:limit]
            out_dir = os.path.join(dst_root, lang, split)
            rows = []
            for fn in files:
                utt_id = os.path.splitext(fn)[0]
                dst = os.path.join(out_dir, utt_id + ".wav")
                jobs.append((ffmpeg, os.path.join(audio_dir, fn), dst, sr))
                rows.append((utt_id, gender_of(utt_id), dst))
            manifests[(lang, split)] = (out_dir, rows)
            print(f"  {lang}/{split}: {len(files)} files")
    return jobs, manifests


def gather_jobs_from_manifest(ffmpeg, manifest_path, sr):
    """Read a build_manifests.py TSV (cols include m4a_path, wav_path) and make jobs."""
    jobs = []
    with open(manifest_path, encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            jobs.append((ffmpeg, row["m4a_path"], row["wav_path"], sr))
    print(f"  {manifest_path}: {len(jobs)} files")
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", default=DEFAULT_SRC)
    ap.add_argument("--dst-root", default=DEFAULT_DST)
    ap.add_argument("--langs", nargs="+", default=["hindi", "sanskrit", "telugu"])
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--sr", type=int, default=TARGET_SR)
    ap.add_argument("--limit", type=int, default=None, help="max files per lang/split (for subsetting/testing)")
    ap.add_argument("--from-manifest", nargs="+", default=None,
                    help="one or more build_manifests.py TSVs; converts exactly the listed utts")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    ffmpeg = find_ffmpeg()
    print(f"ffmpeg: {ffmpeg}")
    print(f"target: mono / 16-bit PCM / {args.sr} Hz")

    manifests = {}
    if args.from_manifest:
        print("Reading manifests:")
        jobs = []
        for mp in args.from_manifest:
            jobs += gather_jobs_from_manifest(ffmpeg, mp, args.sr)
    else:
        print(f"src: {args.src_root}\ndst: {args.dst_root}")
        print("Scanning:")
        jobs, manifests = gather_jobs(ffmpeg, args.src_root, args.dst_root,
                                      args.langs, args.splits, args.sr, args.limit)
    print(f"Total files to process: {len(jobs)} (workers={args.workers})")
    if not jobs:
        return

    ok = skip = fail = 0
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            status, info = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                fails.append(info)
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} | ok={ok} skip={skip} fail={fail}")

    # write manifests (always, so they reflect the full intended set)
    for (lang, split), (out_dir, rows) in manifests.items():
        os.makedirs(out_dir, exist_ok=True)
        mpath = os.path.join(out_dir, "manifest.csv")
        with open(mpath, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "gender", "wav_path"])
            w.writerows(rows)
        print(f"  manifest: {mpath} ({len(rows)} rows)")

    print(f"\nDONE. ok={ok} skip={skip} fail={fail}")
    if fails:
        print("First few failures:")
        for f in fails[:5]:
            print("  ", f)


if __name__ == "__main__":
    main()
