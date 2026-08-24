#!/usr/bin/env python
"""
Memory-safe DSN trainer for the Hindi->Sanskrit UDA pipeline (laptop GPU friendly).

Why this exists: the notebook batches whole utterances and flattens them to frames,
so a batch can be tens of thousands of frames -> OOM on a 6 GB GPU. The DSN is a
*per-frame* feed-forward model (the ±5 splice is already baked into the 1320-d
features), so we can safely CHUNK long utterances into fixed-length pieces. Peak
frames/step is then bounded to  batch_size * chunk_len  regardless of utterance
length, which keeps VRAM flat.

Model / losses are identical to DSN_merged.ipynb (Sec 2.2 / 3.4, Eq. 1-4).

Usage:
    python train_dsn.py                          # full run, defaults tuned for 6 GB
    python train_dsn.py --limit 300 --epochs 1   # quick smoke on a subset
    python train_dsn.py --resume                 # resume from latest checkpoint
"""
import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------
# Model + losses (verbatim from DSN_merged.ipynb)
# ---------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


def grl_alpha_schedule(progress):
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def _mlp_stack(in_dim, hidden_dim, num_layers):
    layers = []
    for i in range(num_layers):
        d_in = in_dim if i == 0 else hidden_dim
        layers += [nn.Linear(d_in, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class PrivateEncoder(nn.Module):
    def __init__(self, input_dim=1320, hidden_dim=512, num_layers=4):
        super().__init__()
        self.net = _mlp_stack(input_dim, hidden_dim, num_layers)

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=1320, hidden_dim=1024, num_layers=6):
        super().__init__()
        self.net = _mlp_stack(input_dim, hidden_dim, num_layers)

    def forward(self, x):
        return self.net(x)


class SenoneClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, num_senones=2472):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_senones),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class SharedDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=1320, num_layers=3):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d_in, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
            d_in = hidden_dim
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DSN(nn.Module):
    def __init__(self, input_dim=1320, shared_dim=1024, private_dim=512, num_senones=2472):
        super().__init__()
        self.private_source = PrivateEncoder(input_dim, private_dim)
        self.private_target = PrivateEncoder(input_dim, private_dim)
        self.shared_encoder = SharedEncoder(input_dim, shared_dim)
        self.senone_classifier = SenoneClassifier(shared_dim, shared_dim, num_senones)
        self.domain_classifier = DomainClassifier(shared_dim, 256)
        self.shared_decoder = SharedDecoder(shared_dim + private_dim, shared_dim, input_dim)

    def forward(self, x, domain='source', mode='train', alpha=1.0):
        private_encoder = self.private_source if domain == 'source' else self.private_target
        private_feat = private_encoder(x)
        shared_feat = self.shared_encoder(x)
        if mode == 'inference':
            return {"shared": shared_feat, "senone_logits": self.senone_classifier(shared_feat)}
        recon = self.shared_decoder(torch.cat([shared_feat, private_feat], dim=1))
        domain_logits = self.domain_classifier(grad_reverse(shared_feat, alpha))
        senone_logits = self.senone_classifier(shared_feat) if domain == 'source' else None
        return {"private": private_feat, "shared": shared_feat, "recon": recon,
                "senone_logits": senone_logits, "domain_logits": domain_logits}


def difference_loss(private_feat, shared_feat):
    if private_feat.size(0) == 0:
        return private_feat.new_zeros(())
    private_feat = private_feat - private_feat.mean(dim=0, keepdim=True)
    shared_feat = shared_feat - shared_feat.mean(dim=0, keepdim=True)
    private_feat = F.normalize(private_feat, dim=1)
    shared_feat = F.normalize(shared_feat, dim=1)
    correlation = private_feat.t() @ shared_feat
    return (correlation ** 2).mean()


def reconstruction_loss(x, x_hat):
    # per-ELEMENT MSE (mean over frames AND feature dims) -> ~O(1).
    # (Eq. 3 sums over the 1320 dims -> ~O(430), which dominated the total loss
    #  and starved senone learning; normalizing lets L_class drive the encoder.)
    if x.size(0) == 0:
        return x.new_zeros(())
    return ((x - x_hat) ** 2).mean()


# ---------------------------------------------------------------------------
# Chunked dataset: bounds frames/step so a 6 GB GPU won't OOM
# ---------------------------------------------------------------------------
class ChunkedNPYDataset(Dataset):
    """One item = a <=chunk_len slice of an utterance. Valid because the model is
    per-frame (splice already baked into the 1320-d features)."""
    def __init__(self, feature_dir, label_dir=None, chunk_len=200, limit=None):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".npy"))
        if limit:
            files = files[:limit]
        self.chunks = []          # (utt_id, start, length)
        for fn in files:
            utt = os.path.splitext(fn)[0]
            T = int(np.load(os.path.join(feature_dir, fn), mmap_mode="r").shape[0])
            for s in range(0, T, chunk_len):
                self.chunks.append((utt, s, min(chunk_len, T - s)))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        utt, s, L = self.chunks[idx]
        feat = np.load(os.path.join(self.feature_dir, utt + ".npy"), mmap_mode="r")[s:s + L]
        feat = torch.as_tensor(np.ascontiguousarray(feat), dtype=torch.float32)
        if self.label_dir:
            lab = np.load(os.path.join(self.label_dir, utt + ".npy"), mmap_mode="r")[s:s + L]
            return feat, torch.as_tensor(np.ascontiguousarray(lab), dtype=torch.long), utt
        return feat, utt


def collate(batch):
    has_labels = len(batch[0]) == 3
    if has_labels:
        feats, labels, ids = zip(*batch)
    else:
        feats, ids = zip(*batch)
    lengths = torch.tensor([f.size(0) for f in feats])
    feats = pad_sequence(feats, batch_first=True)
    T = feats.size(1)
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
    if has_labels:
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return feats, labels, mask, list(ids)
    return feats, mask, list(ids)


def atomic_save(obj, path):
    """Write to a temp file then rename, so a kill mid-write can't truncate the ckpt."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def flatten(x):
    return x.reshape(-1, x.size(-1))


def flatten_mask(m):
    return m.reshape(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", default=r"D:\uda_prep\npy")
    ap.add_argument("--out", default=r"D:\uda_prep\ckpt")
    ap.add_argument("--num-senones", type=int, default=2472)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=24, help="chunks per step")
    ap.add_argument("--chunk-len", type=int, default=400, help="max frames per chunk")
    ap.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--disable-lr-decay", action="store_true",
                    help="hold LR constant (recommended with adam; adam adapts per-param)")
    ap.add_argument("--beta-domain", type=float, default=0.25)
    ap.add_argument("--gamma-diff", type=float, default=0.075)
    ap.add_argument("--delta-recon", type=float, default=0.1)
    # Schedules are FRAME-based (batch-size invariant). Paper: domain loss after
    # 10000 steps and LR decay every 20000 steps at 32-frame batches ->
    # 10000*32=320000 and 20000*32=640000 source frames.
    ap.add_argument("--domain-warmup-frames", type=int, default=320000)
    ap.add_argument("--lr-decay-frames", type=int, default=640000)
    ap.add_argument("--lr-decay-gamma", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=None, help="max utts per set (smoke test)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)
    os.makedirs(args.out, exist_ok=True)

    print("indexing datasets (reading .npy shapes)...", flush=True)
    src_ds = ChunkedNPYDataset(args.npy + r"\hindi\train", args.npy + r"\hindi\train_lab",
                               args.chunk_len, args.limit)
    tgt_ds = ChunkedNPYDataset(args.npy + r"\sanskrit\train", None, args.chunk_len, args.limit)
    print(f"  source chunks: {len(src_ds)} | target chunks: {len(tgt_ds)}", flush=True)

    dl = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
              collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    src_loader = DataLoader(src_ds, **dl)
    tgt_loader = DataLoader(tgt_ds, **dl)

    model = DSN(num_senones=args.num_senones).to(device)
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    print(f"optimizer: {args.optimizer} | base lr: {args.lr} | lr decay: {not args.disable_lr_decay}", flush=True)
    ce = nn.CrossEntropyLoss(ignore_index=-100)

    def set_lr(frames_seen):
        if args.disable_lr_decay:
            return optimizer.param_groups[0]["lr"]
        n_decays = frames_seen // args.lr_decay_frames
        lr = args.lr * (args.lr_decay_gamma ** n_decays)
        for g in optimizer.param_groups:
            g["lr"] = lr
        return lr

    steps_per_epoch = min(len(src_loader), len(tgt_loader))
    total_steps = max(steps_per_epoch * args.epochs, 1)
    start_epoch, global_step, frames_seen = 1, 0, 0

    latest = os.path.join(args.out, "dsn_latest.pth")
    if args.resume and os.path.isfile(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)  # trusted: our own ckpt
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        global_step = ck["global_step"]; frames_seen = ck.get("frames_seen", 0)
        print(f"resumed from epoch {ck['epoch']} (global_step {global_step}, frames {frames_seen})", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.1f}M | steps/epoch: {steps_per_epoch} | total steps: {total_steps}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for (src_x, src_y, src_m, _), (tgt_x, tgt_m, _) in zip(src_loader, tgt_loader):
            src_mf = flatten_mask(src_m)
            tgt_mf = flatten_mask(tgt_m)
            src_xf = flatten(src_x)[src_mf].to(device, non_blocking=True)
            src_yf = src_y.reshape(-1)[src_mf].to(device, non_blocking=True)
            tgt_xf = flatten(tgt_x)[tgt_mf].to(device, non_blocking=True)

            cur_lr = set_lr(frames_seen)
            alpha = grl_alpha_schedule(global_step / total_steps)
            out_s = model(src_xf, domain='source', mode='train', alpha=alpha)
            out_t = model(tgt_xf, domain='target', mode='train', alpha=alpha)

            l_class = ce(out_s["senone_logits"], src_yf)
            dom_s = torch.zeros(src_xf.size(0), dtype=torch.long, device=device)
            dom_t = torch.ones(tgt_xf.size(0), dtype=torch.long, device=device)
            l_sim = ce(out_s["domain_logits"], dom_s) + ce(out_t["domain_logits"], dom_t)
            l_diff = difference_loss(out_s["private"], out_s["shared"]) + \
                     difference_loss(out_t["private"], out_t["shared"])
            l_recon = reconstruction_loss(src_xf, out_s["recon"]) + \
                      reconstruction_loss(tgt_xf, out_t["recon"])

            domain_active = 1.0 if frames_seen >= args.domain_warmup_frames else 0.0
            loss = l_class + args.beta_domain * domain_active * l_sim \
                   + args.gamma_diff * l_diff + args.delta_recon * l_recon

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item(); seen += 1; global_step += 1
            frames_seen += int(src_xf.size(0))
            if seen % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                ips = seen / (time.time() - t0)
                dom = "on" if domain_active else "off"
                print(f"  e{epoch} step {seen}/{steps_per_epoch} | loss {running/seen:.3f} "
                      f"| cls {l_class.item():.3f} | dom {dom} | alpha {alpha:.2f} | lr {cur_lr:.4g} "
                      f"| {ips:.1f} it/s | peak GPU {mem:.2f} GB", flush=True)

        ckpt = {"model": model.state_dict(), "optim": optimizer.state_dict(),
                "epoch": epoch, "global_step": global_step, "frames_seen": frames_seen,
                "num_senones": args.num_senones}
        atomic_save(ckpt, latest)
        atomic_save(ckpt, os.path.join(args.out, f"dsn_epoch{epoch:02d}.pth"))
        print(f"[epoch {epoch}/{args.epochs}] avg loss {running/max(seen,1):.3f} "
              f"| {time.time()-t0:.0f}s | saved {latest}", flush=True)

    print("TRAINING DONE.", flush=True)


if __name__ == "__main__":
    main()
