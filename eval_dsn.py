#!/usr/bin/env python
"""
Step-1 evaluation of the trained DSN (no Kaldi needed):
  1. Domain accuracy (Table 2 style): can the domain classifier still tell Hindi
     from Sanskrit at the shared-encoder output?  LOWER = more domain-invariant
     = UDA worked. (~50% is chance for a balanced 2-way split.)
  2. Frame-level senone accuracy on Hindi (sanity that the acoustic model learned;
     note this is on training data, so it's an optimistic upper bound).
  3. Inference on Sanskrit test -> per-utterance senone sequences (saved).

WER is NOT computed here (needs the Kaldi WFST decoder, Step 2).
"""
import argparse, os, numpy as np, torch
from torch.utils.data import DataLoader
from train_dsn import DSN, ChunkedNPYDataset, collate, flatten, flatten_mask


@torch.no_grad()
def domain_accuracy(model, loader, domain_label, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, batch in enumerate(loader):
        feats, mask = batch[0], batch[-2]
        xf = flatten(feats)[flatten_mask(mask)].to(device)
        if xf.numel() == 0:
            continue
        logits = model.domain_classifier(model.shared_encoder(xf))
        correct += (logits.argmax(1) == domain_label).sum().item()
        total += xf.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return correct / max(total, 1), total


@torch.no_grad()
def frame_accuracy(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (feats, labels, mask, _) in enumerate(loader):
        mf = flatten_mask(mask)
        xf = flatten(feats)[mf].to(device)
        yf = labels.reshape(-1)[mf].to(device)
        preds = model.senone_classifier(model.shared_encoder(xf)).argmax(1)
        correct += (preds == yf).sum().item()
        total += xf.size(0)
        if max_batches and i + 1 >= max_batches:
            break
    return correct / max(total, 1), total


@torch.no_grad()
def infer_sanskrit_test(model, npy, device, out_path):
    """Per-utterance senone argmax on full (un-chunked) utterances."""
    model.eval()
    d = os.path.join(npy, "sanskrit", "test")
    preds = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".npy")):
        utt = os.path.splitext(fn)[0]
        feat = torch.as_tensor(np.load(os.path.join(d, fn)).astype(np.float32)).to(device)
        logits = model.senone_classifier(model.shared_encoder(feat))
        preds[utt] = logits.argmax(1).cpu().numpy().astype(np.int16)
    np.savez_compressed(out_path, **preds)
    return len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\uda_prep\ckpt\dsn_latest.pth")
    ap.add_argument("--npy", default=r"D:\uda_prep\npy")
    ap.add_argument("--num-senones", type=int, default=2472)
    ap.add_argument("--limit", type=int, default=500, help="utts per set for the accuracy metrics")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--chunk-len", type=int, default=400)
    ap.add_argument("--out", default=r"D:\uda_prep\pred\sanskrit_test_senones.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DSN(num_senones=args.num_senones).to(device)
    model.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt} (epoch {ck['epoch']}) on {device}", flush=True)

    def loader(sub, lab=None):
        ds = ChunkedNPYDataset(os.path.join(args.npy, *sub.split("/")),
                               os.path.join(args.npy, *lab.split("/")) if lab else None,
                               args.chunk_len, args.limit)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=4, collate_fn=collate)

    hi = loader("hindi/train", "hindi/train_lab")
    sa = loader("sanskrit/test")

    da_h, nh = domain_accuracy(model, hi, 0, device)   # Hindi frames, true label 0
    da_s, ns = domain_accuracy(model, sa, 1, device)   # Sanskrit frames, true label 1
    fa, nf = frame_accuracy(model, hi, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_utt = infer_sanskrit_test(model, args.npy, device, args.out)

    print("\n================ DSN Step-1 evaluation ================")
    print(f"Domain accuracy  Hindi  (n={nh:>8}): {da_h*100:5.2f}%   (lower = more invariant)")
    print(f"Domain accuracy  Sanskrit (n={ns:>6}): {da_s*100:5.2f}%   (lower = more invariant)")
    print(f"Frame senone acc Hindi  (n={nf:>8}): {fa*100:5.2f}%   (train data, optimistic)")
    print(f"Sanskrit-test senone predictions saved: {n_utt} utts -> {args.out}")
    print("=======================================================")


if __name__ == "__main__":
    main()
