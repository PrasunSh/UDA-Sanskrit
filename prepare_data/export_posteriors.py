#!/usr/bin/env python
"""
Export DSN acoustic scores for the Sanskrit test set as a Kaldi matrix ark that
latgen-faster-mapped can decode.

Per frame we write  log P(senone|x) - log P(senone)  (pseudo log-likelihood,
Sec 3.5): the senone posteriors from the shared-encoder->senone-classifier path,
minus the log senone priors estimated from the Hindi training alignments.

Columns are pdf-ids 0..num_senones-1, matching exp/tri3/final.mdl.

Output: <decode>/loglikes.ark  (read in-container as ark:decode/loglikes.ark)
"""
import argparse, os, sys, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from train_dsn import DSN
import kaldiio


@torch.no_grad()
def log_priors_from_labels(lab_dir, num_senones, device, floor=1e-8):
    counts = np.zeros(num_senones, dtype=np.int64)
    files = glob.glob(os.path.join(lab_dir, "*.npy"))
    for i, f in enumerate(files):
        lab = np.load(f)
        counts += np.bincount(lab, minlength=num_senones)[:num_senones]
    p = counts / max(counts.sum(), 1)
    lp = np.log(p + floor).astype(np.float32)
    return torch.from_numpy(lp).to(device), int(counts.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=r"D:\uda_prep\ckpt_adam\dsn_latest.pth")
    ap.add_argument("--npy", default=r"D:\uda_prep\npy")
    ap.add_argument("--num-senones", type=int, default=2472)
    ap.add_argument("--out", default=r"D:\uda_prep\kaldi\decode\loglikes.ark")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DSN(num_senones=args.num_senones).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {args.ckpt} (epoch {ck['epoch']})", flush=True)

    log_prior, nframes = log_priors_from_labels(
        os.path.join(args.npy, "hindi", "train_lab"), args.num_senones, device)
    print(f"log-priors from {nframes} Hindi frames", flush=True)

    test_dir = os.path.join(args.npy, "sanskrit", "test")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d = {}
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(test_dir, "*.npy"))):
            utt = os.path.splitext(os.path.basename(f))[0]
            x = torch.as_tensor(np.load(f).astype(np.float32)).to(device)
            logits = model.senone_classifier(model.shared_encoder(x))
            loglike = (torch.log_softmax(logits, dim=1) - log_prior).cpu().numpy().astype(np.float32)
            d[utt] = np.ascontiguousarray(loglike)
    kaldiio.save_ark(args.out, d)
    print(f"wrote {len(d)} utts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
