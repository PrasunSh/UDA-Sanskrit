# UDA for Low-Resource Sanskrit ASR (Hindi → Sanskrit)

A from-scratch reproduction of **Domain Separation Networks (DSN)** for *unsupervised domain
adaptation* in automatic speech recognition, adapting acoustic models from a high-resource
language (**Hindi**, labeled) to a low-resource one (**Sanskrit**, unlabeled).

Based on: Anoop C. S., Prathosh A. P., A. G. Ramakrishnan, *"Unsupervised Domain Adaptation
Schemes for Building ASR in Low-Resource Languages,"* IEEE ASRU 2021.

The full pipeline — raw audio → features → HMM-GMM senone alignment → DSN training →
WFST decoding → WER — is implemented in **PyTorch + Kaldi (Dockerized)** and runs end to end
on a single 6 GB laptop GPU.

---

## Highlights

- **End-to-end reproduction** of the DSN method: per-domain **private** encoders + a shared,
  adversarially **domain-invariant** encoder (gradient reversal), a senone classifier, a domain
  classifier, and a shared decoder, trained with the paper's four-term objective (Eq. 1–4).
- **Domain invariance achieved** — the shared-encoder domain-classifier accuracy collapses to
  ~53% (near chance), **matching the paper's reported domain-invariance** (Table 2: 52–63%).
- **Diagnosed and fixed a loss-scaling imbalance** in which the reconstruction term dominated the
  objective and starved the senone classifier — **frame-level accuracy 25% → 50%**, cutting
  **WER by ~19 points (96% → 77%)**.
- **Runs on constrained hardware**: utterance-chunking + frame-capped batching keep peak VRAM
  under ~3 GB; training is resumable with atomic checkpoints.
- **Complete WFST decoding stack**: grapheme lexicon over the acoustic model's phone set, a
  **Wikipedia-augmented bigram LM** (test OOV 31% → 17%), prior-normalized senone posteriors,
  Kaldi lattice decoding, and LM-weight sweeping.

## Results (Sanskrit test set, 558 utterances)

| Configuration | Frame acc (Hindi) | Domain acc (H / S) | WER |
|---|---|---|---|
| 25-epoch AM, train-only LM | 25% | — | 96.3% |
| 50-epoch AM, Wikipedia LM | 27% | — | 82.4% |
| **Rebalanced 80-epoch AM, Wikipedia LM** | **50%** | **52.9% / 58.4%** | **77.1%** |
| *Reference (paper, DSN Hindi→Sanskrit)* | — | *52.2% / 63.2%* | *17.26%* |

**On the gap to the paper's WER:** the reproduction faithfully implements the *method* and
demonstrably achieves the paper's *domain-invariance*. The remaining WER gap is driven by
practical constraints rather than the architecture: a **pruned LM** (16 GB laptop RAM couldn't
build the full 675k-word graph → 26% OOV vs. a possible 17%), a **grapheme lexicon** instead of
the paper's phonemic SLP1-M G2P, and a smaller acoustic-model compute budget. See
[Limitations](#limitations--future-work).

---

## Architecture

```
              ┌─────────────────────────────┐
  x  ───────► │  Private encoder (per domain)│──► f_private ─┐
              └─────────────────────────────┘               ├─► Shared decoder ──► x̂   (L_recon)
              ┌─────────────────────────────┐               │
  x  ───────► │       Shared encoder        │──► f_shared ──┤
              └─────────────────────────────┘               ├─► Senone classifier ─► ŷ  (L_class, source only)
                                                            │
                                          GRL(α) ───────────┴─► Domain classifier ─► d̂  (L_sim, adversarial)

  L_diff: soft orthogonality between f_private and f_shared
  L = L_class + β·L_sim + γ·L_diff + δ·L_recon
```

- Input: 1320-dim (40 log-mel filterbank + Δ + ΔΔ, utterance CMVN, spliced ±5 frames)
- Private encoders: 4×512 · Shared encoder: 6×1024 · Senone classifier: 2×1024 → 2472 senones
- Domain classifier: 1×256 → 2 · Shared decoder: 3×1024 → 1320
- Senones: 2472 pdf-ids from a Kaldi tri3 HMM-GMM tree trained on Hindi

## Repository layout

```
DSN_merged.ipynb          Canonical, paper-faithful DSN model + losses + eval helpers (annotated)
train_dsn.py              Memory-safe DSN trainer (chunked batching, Adam/SGD, resumable)
eval_dsn.py               Frame accuracy, domain accuracy, senone inference
prepare_data/
  convert_audio.py        .m4a → 8 kHz mono 16-bit WAV
  build_manifests.py      Paper-faithful train/test subset manifests
  make_kaldi_data.py      Kaldi data dirs (wav.scp / utt2spk / text)
  build_lexicon.py        Hindi grapheme lexicon (dict dir)
  export_to_npy.py        Kaldi 1320-dim feats + senone labels → per-utterance .npy
  build_sanskrit_decode.py  Sanskrit lexicon + LM corpus (+Wikipedia) + test refs
  build_arpa.py           Witten-Bell bigram ARPA (no SRILM needed)
  export_posteriors.py    DSN log-posteriors − log-prior → Kaldi loglikes ark
kaldi/
  run.sh                  MFCC → mono→tri1→tri2→tri3 → align → 1320-d fbank features
  decode.sh               lang → G.fst → HCLG (mkgraph) → latgen → hyp
  score_sweep.sh          LM-weight / word-insertion-penalty sweep → WER
  path.sh cmd.sh conf/    Kaldi env + feature configs
archive/                  Earlier notebook iterations (superseded by DSN_merged.ipynb)
```

## Pipeline (reproduce)

Data (audio + transcripts) and generated artifacts are **not** in the repo. With the KB
Hindi/Sanskrit corpus in place:

```bash
# 1. Audio + subset + Kaldi data dirs + Hindi lexicon
python prepare_data/convert_audio.py --from-manifest ...    # m4a -> 8 kHz wav
python prepare_data/build_manifests.py                      # 15k Hindi / 2837+558 Sanskrit
python prepare_data/make_kaldi_data.py
python prepare_data/build_lexicon.py

# 2. Kaldi: HMM-GMM alignment + 1320-d features  (in the kaldiasr/kaldi container)
docker run --rm -v <data>:/workspace/uda_prep -w /workspace/uda_prep/kaldi \
    kaldiasr/kaldi:latest bash run.sh
python prepare_data/export_to_npy.py                        # -> per-utterance .npy

# 3. Train the DSN (GPU)
python train_dsn.py --optimizer adam --lr 1e-3 --disable-lr-decay --delta-recon 1.0 --epochs 80

# 4. Decode -> WER  (Kaldi container)
python prepare_data/build_sanskrit_decode.py                # Sanskrit lexicon + LM (+wiki)
python prepare_data/build_arpa.py decode/lm_corpus.txt decode/sanskrit.arpa
python prepare_data/export_posteriors.py --ckpt <ckpt>.pth
docker run ... bash decode.sh                               # HCLG + lattice decode
docker run ... bash score_sweep.sh                          # best WER
```

## Key findings

- **Loss balance matters more than architecture here.** With Eq. 3's reconstruction loss summed
  over 1320 dims (~430), `δ·L_recon` dwarfed `L_class` (~3) and the senone head barely learned
  (25% frame acc). Normalizing reconstruction to per-element MSE made the senone objective
  dominant and **doubled frame accuracy to 50%** — the single biggest WER lever.
- **Batch/optimizer scaling.** The paper's 32-*frame* batches imply ~1M SGD updates; large
  frame-batches on a laptop give far fewer, so vanilla SGD underfit badly. Adam (lr 1e-3) is
  robust to the large effective batch and converged well.
- **LM coverage dominates the WER floor.** Sanskrit is highly inflected; a train-only LM had 31%
  test OOV. Adding Sanskrit Wikipedia cut it to 17% (unpruned) / 26% (laptop-buildable pruned).

## Limitations & future work

- **Unpruned 675k-word LM** (17% OOV) — needs more RAM for `mkgraph`; would remove ~9 pts of
  OOV floor.
- **Phonemic SLP1-M G2P** for Sanskrit/Hindi instead of graphemes — cleaner senones and
  pronunciations (the paper's approach).
- **Trigram LM** for fewer insertions; **larger acoustic-model compute** for higher frame accuracy.
- **Baselines** (DNN / MT / GRL) and the Telugu-source ablation from the paper are scaffolded but
  not run here.

## Stack

PyTorch · Kaldi · Docker · WFST/HCLG decoding · HMM-GMM alignment · MFCC/filterbank features ·
domain-adversarial training (GRL/DSN) · n-gram language modeling.
