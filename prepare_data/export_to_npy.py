#!/usr/bin/env python
"""
Export Kaldi recipe outputs to the per-utterance .npy files that the notebook's
LazyNPYDataset consumes. Runs on the HOST (reads the binary .ark files directly
via kaldiio, so no container paths are needed).

Reads:
    <kaldi>/feats1320/<set>/feats1320.ark        (1320-d features per utt)
    <kaldi>/exp/tri3_ali/pdf.txt                 (per-frame senone ids, Hindi)

Writes:
    <out>/hindi/train/<utt>.npy       features [T,1320] float32
    <out>/hindi/train_lab/<utt>.npy   senone labels [T] int64        (source labels)
    <out>/sanskrit/train/<utt>.npy    features [T,1320]               (unlabeled)
    <out>/sanskrit/test/<utt>.npy     features [T,1320]               (unlabeled)

For Hindi, feature and label frame counts must match (both 25/10 ms framing).
Any small mismatch is trimmed to the common length with a warning; large
mismatches are reported and skipped.
"""
import argparse
import os
import numpy as np
import kaldiio

SETS = {
    # set_name        (lang,      split,   labeled)
    "hindi_train":    ("hindi",    "train", True),
    "sanskrit_train": ("sanskrit", "train", False),
    "sanskrit_test":  ("sanskrit", "test",  False),
}


def load_pdf(pdf_path):
    """pdf.txt: '<utt> id id id ...' per line -> {utt: np.int64[T]}."""
    labels = {}
    with open(pdf_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            labels[parts[0]] = np.asarray(parts[1:], dtype=np.int64)
    return labels


def export_set(name, kaldi_root, out_root, labels, max_trim, feat_dtype):
    lang, split, labeled = SETS[name]
    ark = os.path.join(kaldi_root, "feats1320", name, "feats1320.ark")
    if not os.path.isfile(ark):
        print(f"  [{name}] MISSING {ark} -- run run.sh stage 5 first."); return
    feat_dir = os.path.join(out_root, lang, split)
    lab_dir = os.path.join(out_root, lang, split + "_lab") if labeled else None
    os.makedirs(feat_dir, exist_ok=True)
    if lab_dir:
        os.makedirs(lab_dir, exist_ok=True)

    n, n_lab, skipped, trimmed = 0, 0, 0, 0
    for utt, feat in kaldiio.load_ark(ark):
        feat = np.asarray(feat, dtype=feat_dtype)
        if labeled:
            lab = labels.get(utt)
            if lab is None:
                skipped += 1
                continue
            if len(lab) != feat.shape[0]:
                diff = abs(len(lab) - feat.shape[0])
                if diff > max_trim:
                    print(f"    skip {utt}: T mismatch feat={feat.shape[0]} lab={len(lab)}")
                    skipped += 1
                    continue
                T = min(len(lab), feat.shape[0])
                feat, lab = feat[:T], lab[:T]
                trimmed += 1
            np.save(os.path.join(lab_dir, utt + ".npy"), lab)
            n_lab += 1
        np.save(os.path.join(feat_dir, utt + ".npy"), feat)
        n += 1
    msg = f"  [{name}] feats={n} -> {feat_dir}"
    if labeled:
        msg += f" | labels={n_lab} (trimmed={trimmed}, skipped={skipped})"
    print(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaldi-root", default=r"D:\uda_prep\kaldi")
    ap.add_argument("--out", default=r"D:\uda_prep\npy")
    ap.add_argument("--pdf", default=None, help="default: <kaldi-root>/exp/tri3_ali/pdf.txt")
    ap.add_argument("--max-trim", type=int, default=3,
                    help="max |feat_T - label_T| to trim instead of skip")
    ap.add_argument("--feat-dtype", default="float16", choices=["float16", "float32"],
                    help="on-disk feature dtype (float16 halves size; cast to float32 at load)")
    args = ap.parse_args()
    feat_dtype = np.float16 if args.feat_dtype == "float16" else np.float32

    pdf_path = args.pdf or os.path.join(args.kaldi_root, "exp", "tri3_ali", "pdf.txt")
    labels = load_pdf(pdf_path) if os.path.isfile(pdf_path) else {}
    if not labels:
        print(f"WARNING: no labels loaded from {pdf_path} (Hindi export will be skipped).")

    print(f"kaldi: {args.kaldi_root}\nout:   {args.out}\nfeat dtype: {args.feat_dtype}")
    for name in SETS:
        export_set(name, args.kaldi_root, args.out, labels, args.max_trim, feat_dtype)
    print("\nDone. Point the notebook at:")
    print(r"  hindi_feats    = <out>\hindi\train")
    print(r"  hindi_labels   = <out>\hindi\train_lab")
    print(r"  sanskrit_train = <out>\sanskrit\train")
    print(r"  sanskrit_test  = <out>\sanskrit\test")


if __name__ == "__main__":
    main()
