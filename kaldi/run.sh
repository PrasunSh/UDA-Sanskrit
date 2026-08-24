#!/usr/bin/env bash
# DSN data-prep recipe (runs INSIDE the kaldiasr/kaldi container).
#
# Produces, for the paper's Hindi->Sanskrit DSN:
#   - per-frame senone (pdf-id) alignments for Hindi   -> exp/tri3_ali/pdf.txt
#   - number of senones                                -> exp/tri3/hmm_info.txt
#   - 1320-dim DSN features (40 fbank + delta + accel, utt-CMVN, splice +-5)
#     for hindi_train / sanskrit_train / sanskrit_test -> feats1320/<set>/feats1320.ark
#
# Run it with the repo working dir mounted, e.g.:
#   docker run --rm -it -v D:\uda_prep:/workspace/uda_prep -w /workspace/uda_prep/kaldi \
#       kaldiasr/kaldi:latest ./run.sh
set -euo pipefail

export KALDI_ROOT=/opt/kaldi
cd "$(dirname "$0")"

# link the standard utils/ and steps/ from the wsj recipe (copy if symlinks
# aren't supported on the mounted volume)
for d in utils steps; do
  if [ ! -e "$d" ]; then
    ln -s "$KALDI_ROOT/egs/wsj/s5/$d" "$d" 2>/dev/null || cp -r "$KALDI_ROOT/egs/wsj/s5/$d" "$d"
  fi
done
. ./path.sh
. ./cmd.sh

nj="${NJ:-8}"                       # parallel jobs (override with NJ=... ./run.sh)
leaves="${LEAVES:-3080}"            # target #senones (paper: 3080)
totgauss="${TOTGAUSS:-30000}"
stage="${STAGE:-0}"

echo "== nj=$nj leaves=$leaves totgauss=$totgauss stage=$stage =="

# ---------------------------------------------------------------------------
# 0) validate data dirs
# ---------------------------------------------------------------------------
if [ $stage -le 0 ]; then
  for x in hindi_train sanskrit_train sanskrit_test; do
    utils/utt2spk_to_spk2utt.pl data/$x/utt2spk > data/$x/spk2utt
    utils/fix_data_dir.sh data/$x
    # --non-print true: allow non-ASCII (Devanagari) transcripts
    utils/validate_data_dir.sh --no-feats --non-print true data/$x || true
  done
fi

# ---------------------------------------------------------------------------
# 1) lang dir from the grapheme lexicon
# ---------------------------------------------------------------------------
if [ $stage -le 1 ]; then
  utils/prepare_lang.sh dict_hindi "<UNK>" data/local/lang_tmp data/lang_hindi
fi

# ---------------------------------------------------------------------------
# 2) MFCC (+utt CMVN) for GMM training on Hindi
# ---------------------------------------------------------------------------
if [ $stage -le 2 ]; then
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc.conf \
      data/hindi_train exp/make_mfcc/hindi_train mfcc
  steps/compute_cmvn_stats.sh data/hindi_train exp/make_mfcc/hindi_train mfcc
  utils/fix_data_dir.sh data/hindi_train
fi

# ---------------------------------------------------------------------------
# 3) monophone -> tri1 (deltas) -> tri2 (LDA+MLLT) -> tri3 (bigger, ~leaves senones)
# ---------------------------------------------------------------------------
if [ $stage -le 3 ]; then
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
      data/hindi_train data/lang_hindi exp/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/hindi_train data/lang_hindi exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 \
      data/hindi_train data/lang_hindi exp/mono_ali exp/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/hindi_train data/lang_hindi exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" 2500 15000 \
      data/hindi_train data/lang_hindi exp/tri1_ali exp/tri2
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/hindi_train data/lang_hindi exp/tri2 exp/tri2_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" $leaves $totgauss \
      data/hindi_train data/lang_hindi exp/tri2_ali exp/tri3
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
      data/hindi_train data/lang_hindi exp/tri3 exp/tri3_ali
fi

# ---------------------------------------------------------------------------
# 4) export per-frame senone (pdf) ids for Hindi + report #senones
# ---------------------------------------------------------------------------
if [ $stage -le 4 ]; then
  ali-to-pdf exp/tri3/final.mdl "ark:gunzip -c exp/tri3_ali/ali.*.gz|" \
      ark,t:exp/tri3_ali/pdf.txt
  hmm-info exp/tri3/final.mdl | tee exp/tri3/hmm_info.txt
  echo "num-senones = $(grep 'number of pdfs' exp/tri3/hmm_info.txt | awk '{print $NF}')"
fi

# ---------------------------------------------------------------------------
# 5) 1320-dim DSN features for all three sets
#    fbank(40) -> add-deltas(120) -> utt CMVN -> splice +-5 -> 1320
# ---------------------------------------------------------------------------
if [ $stage -le 5 ]; then
  for x in hindi_train sanskrit_train sanskrit_test; do
    # Build the fbank work dir by hand with only what feature extraction needs.
    # (copy_data_dir.sh validates the copy, and the Devanagari text trips the
    #  non-printable check; feats don't need text at all.)
    rm -rf data/${x}_fb
    mkdir -p data/${x}_fb
    cp data/$x/wav.scp data/$x/utt2spk data/$x/spk2utt data/${x}_fb/
    steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank-config conf/fbank.conf \
        data/${x}_fb exp/make_fbank/$x fbank
    utils/fix_data_dir.sh data/${x}_fb

    out=feats1320/$x
    mkdir -p $out
    # 120-d = fbank + delta + accel
    add-deltas scp:data/${x}_fb/feats.scp ark,scp:$out/delta.ark,$out/delta.scp
    # utterance-level CMVN on the 120-d (spk2utt is identity)
    compute-cmvn-stats --spk2utt=ark:data/${x}_fb/spk2utt \
        scp:$out/delta.scp ark,scp:$out/cmvn.ark,$out/cmvn.scp
    # apply CMVN then splice +-5 -> 1320
    apply-cmvn --utt2spk=ark:data/${x}_fb/utt2spk scp:$out/cmvn.scp scp:$out/delta.scp ark:- | \
      splice-feats --left-context=5 --right-context=5 ark:- \
        ark,scp:$out/feats1320.ark,$out/feats1320.scp
    echo "  $x: $(feat-to-dim scp:$out/feats1320.scp - 2>/dev/null) -dim feats -> $out/feats1320.ark"
  done
fi

echo "DONE. Next (on host): export_to_npy.py"
