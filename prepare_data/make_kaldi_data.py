#!/usr/bin/env python
"""
Build Kaldi data directories from our subset manifests.

For each set we emit a standard Kaldi data dir with C-sorted files:
    wav.scp   : <utt_id> <container_wav_path>
    utt2spk   : <utt_id> <utt_id>          (identity -> utterance-level CMVN, per paper Sec 3.2)
    spk2utt   : <utt_id> <utt_id>
    text      : <utt_id> <transcript>      (Hindi = alignment target; Sanskrit-test = WER ref)

Paths in wav.scp use the CONTAINER mount point, not the Windows path, because the
recipe runs inside the kaldiasr/kaldi container with D:\\uda_prep mounted there:
    docker run -v D:\\uda_prep:/workspace/uda_prep ...

Utterance-level CMVN: the paper normalizes per utterance, so each utt is its own
"speaker" (utt2spk/spk2utt are identity maps). The real speaker id (middle field of
the filename) is intentionally NOT used for CMVN grouping.
"""
import argparse
import csv
import os

MANIFESTS = {
    # set_name        (manifest file,           write_text)
    "hindi_train":    ("hindi_train.tsv",    True),   # source: needed for alignment
    "sanskrit_train": ("sanskrit_train.tsv", True),   # target: features only (text harmless)
    "sanskrit_test":  ("sanskrit_test.tsv",  True),   # target test: text = WER reference
}


def to_container_path(win_path, win_root, mount):
    """D:\\uda_prep\\wav\\hindi\\train\\x.wav -> <mount>/wav/hindi/train/x.wav"""
    rel = os.path.relpath(win_path, win_root)
    return mount.rstrip("/") + "/" + rel.replace("\\", "/")


def write_sorted(path, lines):
    lines = sorted(lines)                       # C/byte order (utt_ids are ASCII)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def build_set(name, man_path, write_text, win_root, mount, out_root):
    with open(man_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    d = os.path.join(out_root, name)
    os.makedirs(d, exist_ok=True)

    wav, u2s, s2u, txt = [], [], [], []
    for r in rows:
        utt = r["utt_id"]
        wav.append(f"{utt} {to_container_path(r['wav_path'], win_root, mount)}")
        u2s.append(f"{utt} {utt}")
        s2u.append(f"{utt} {utt}")
        if write_text:
            txt.append(f"{utt} {r['transcript'].strip()}")

    write_sorted(os.path.join(d, "wav.scp"), wav)
    write_sorted(os.path.join(d, "utt2spk"), u2s)
    write_sorted(os.path.join(d, "spk2utt"), s2u)
    if write_text:
        write_sorted(os.path.join(d, "text"), txt)
    print(f"  {name}: {len(rows)} utts -> {d}"
          + ("  (+text)" if write_text else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest-dir", default=r"D:\uda_prep\manifests")
    ap.add_argument("--win-root", default=r"D:\uda_prep", help="Windows dir mounted into the container")
    ap.add_argument("--mount", default="/workspace/uda_prep", help="container mount point for --win-root")
    ap.add_argument("--out", default=r"D:\uda_prep\kaldi\data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"mount: {args.win_root}  ->  {args.mount}")
    print(f"out:   {args.out}")
    for name, (mf, wt) in MANIFESTS.items():
        build_set(name, os.path.join(args.manifest_dir, mf), wt,
                  args.win_root, args.mount, args.out)
    print("\nData dirs ready. Inside the container, run `utils/fix_data_dir.sh` on each before use.")


if __name__ == "__main__":
    main()
