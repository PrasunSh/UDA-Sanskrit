#!/usr/bin/env python
"""
Select the paper-faithful subset (Sec 3.1) and write manifests pairing each
utterance's .m4a, its target .wav path, and its transcript.

Paper subsets:
    Hindi (source)   : 15,000 train utts               -> source, labeled (senone labels via alignment)
    Sanskrit (target): 3,395 utts -> 2,837 train + 558 test  (random split, like the paper)
    Telugu (optional): same size as Hindi if enabled    -> alternative source (ablation 4.1.4)

Inputs:
    audio       : <audio_root>/<lang>/train/audio/<utt_id>.m4a
    transcripts : <trans_root>/<lang>/train/transcription_n2w.txt   (tab: "<utt_id>.m4a\\t<text>")

Outputs (under <out_root>/manifests/):
    hindi_train.tsv, sanskrit_train.tsv, sanskrit_test.tsv [, telugu_train.tsv]
    columns: utt_id \\t gender \\t split \\t m4a_path \\t wav_path \\t transcript

The wav_path points at where convert_audio.py (--from-manifest) will write the 8 kHz WAV:
    <out_root>/wav/<lang>/<split>/<utt_id>.wav
"""
import argparse
import csv
import os
import random

AUDIO_ROOT = r"D:\train_audio\kb_data_clean_m4a"
TRANS_ROOT = r"D:\transcripts_n2w\kb_data_clean_m4a"
OUT_ROOT = r"D:\uda_prep"


def gender_of(utt_id):
    tail = utt_id.rsplit("-", 1)[-1].lower()
    return {"m": "male", "f": "female"}.get(tail, "unknown")


def load_transcripts(path):
    """Returns {utt_id: transcript}. First column may carry a .m4a suffix."""
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(None, 1)          # fallback: split on first whitespace
            if len(parts) < 2:
                continue
            utt = os.path.splitext(parts[0].strip())[0]
            d[utt] = parts[1].strip()
    return d


def available_ids(lang):
    audio_dir = os.path.join(AUDIO_ROOT, lang, "train", "audio")
    return sorted(os.path.splitext(f)[0] for f in os.listdir(audio_dir) if f.lower().endswith(".m4a"))


def write_manifest(out_dir, name, lang, split, ids, trans):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    audio_dir = os.path.join(AUDIO_ROOT, lang, "train", "audio")
    wav_dir = os.path.join(OUT_ROOT, "wav", lang, split)
    n_missing = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["utt_id", "gender", "split", "m4a_path", "wav_path", "transcript"])
        for utt in ids:
            if utt not in trans:
                n_missing += 1
                continue
            w.writerow([utt, gender_of(utt), split,
                        os.path.join(audio_dir, utt + ".m4a"),
                        os.path.join(wav_dir, utt + ".wav"),
                        trans[utt]])
    print(f"  wrote {path}  ({len(ids) - n_missing} rows"
          + (f", {n_missing} skipped: no transcript)" if n_missing else ")"))
    return path


def main():
    global AUDIO_ROOT, TRANS_ROOT, OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-root", default=AUDIO_ROOT)
    ap.add_argument("--trans-root", default=TRANS_ROOT)
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--n-hindi", type=int, default=15000, help="source (Hindi) train utts")
    ap.add_argument("--n-sanskrit-train", type=int, default=2837, help="target train utts")
    ap.add_argument("--n-sanskrit-test", type=int, default=558, help="target test utts")
    ap.add_argument("--include-telugu", action="store_true", help="also build a Telugu source manifest")
    ap.add_argument("--n-telugu", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    AUDIO_ROOT, TRANS_ROOT, OUT_ROOT = args.audio_root, args.trans_root, args.out_root
    rng = random.Random(args.seed)
    man_dir = os.path.join(OUT_ROOT, "manifests")
    print(f"out: {OUT_ROOT}\nseed: {args.seed}\n")

    # --- Hindi source ---
    print("Hindi (source):")
    hi_ids = available_ids("hindi")
    hi_tr = load_transcripts(os.path.join(TRANS_ROOT, "hindi", "train", "transcription_n2w.txt"))
    hi_ids = [u for u in hi_ids if u in hi_tr]
    rng.shuffle(hi_ids)
    write_manifest(man_dir, "hindi_train.tsv", "hindi", "train", hi_ids[:args.n_hindi], hi_tr)

    # --- Sanskrit target: sample pool, then split train/test ---
    print("Sanskrit (target):")
    sa_ids = available_ids("sanskrit")
    sa_tr = load_transcripts(os.path.join(TRANS_ROOT, "sanskrit", "train", "transcription_n2w.txt"))
    sa_ids = [u for u in sa_ids if u in sa_tr]
    rng.shuffle(sa_ids)
    need = args.n_sanskrit_train + args.n_sanskrit_test
    pool = sa_ids[:need]
    sa_train, sa_test = pool[:args.n_sanskrit_train], pool[args.n_sanskrit_train:need]
    write_manifest(man_dir, "sanskrit_train.tsv", "sanskrit", "train", sa_train, sa_tr)
    write_manifest(man_dir, "sanskrit_test.tsv", "sanskrit", "test", sa_test, sa_tr)

    # --- Telugu (optional) ---
    if args.include_telugu:
        print("Telugu (optional source):")
        te_ids = available_ids("telugu")
        te_tr = load_transcripts(os.path.join(TRANS_ROOT, "telugu", "train", "transcription_n2w.txt"))
        te_ids = [u for u in te_ids if u in te_tr]
        rng.shuffle(te_ids)
        write_manifest(man_dir, "telugu_train.tsv", "telugu", "train", te_ids[:args.n_telugu], te_tr)

    print("\nDone. Next: convert_audio.py --from-manifest for each manifest.")


if __name__ == "__main__":
    main()
