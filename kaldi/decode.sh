#!/usr/bin/env bash
# WFST decoding of the DSN senone scores -> Sanskrit words -> WER.
# Run in the kaldiasr/kaldi container:
#   docker run --rm -v D:\uda_prep:/workspace/uda_prep -w /workspace/uda_prep/kaldi \
#       kaldiasr/kaldi:latest bash decode.sh
set -euo pipefail
export KALDI_ROOT=/opt/kaldi
cd "$(dirname "$0")"
for d in utils steps; do
  [ -e "$d" ] || { ln -s "$KALDI_ROOT/egs/wsj/s5/$d" "$d" 2>/dev/null || cp -r "$KALDI_ROOT/egs/wsj/s5/$d" "$d"; }
done
. ./path.sh
. ./cmd.sh

acwt="${ACWT:-0.1}"
stage="${STAGE:-0}"
end="${END:-99}"
run() { [ $stage -le "$1" ] && [ $end -ge "$1" ]; }

# 1) Sanskrit lang dir, forcing the SAME phone symbol table as the Hindi AM
if run 1; then
  utils/prepare_lang.sh --phone-symbol-table data/lang_hindi/phones.txt \
      dict_sanskrit "<UNK>" data/local/lang_sa_tmp data/lang_sanskrit
fi

# 2) G.fst from our bigram ARPA
if run 2; then
  rm -rf data/lang_sanskrit_test
  cp -r data/lang_sanskrit data/lang_sanskrit_test
  arpa2fst --disambig-symbol='#0' --read-symbol-table=data/lang_sanskrit/words.txt \
      decode/sanskrit.arpa data/lang_sanskrit_test/G.fst
  fstisstochastic data/lang_sanskrit_test/G.fst || echo "  (G not stochastic -- ok)"
  utils/validate_lang.pl --skip-determinization-check data/lang_sanskrit_test || true
fi

# 3) HCLG graph (composes H C L G using the Hindi tri3 model)
if run 3; then
  utils/mkgraph.sh data/lang_sanskrit_test exp/tri3 exp/tri3/graph_sa
fi

# 4) decode DSN loglikes -> lattices -> best path -> hyp words
if run 4; then
  mkdir -p decode/out
  latgen-faster-mapped --acoustic-scale=$acwt --beam=13.0 --lattice-beam=6.0 --max-active=7000 \
      --word-symbol-table=data/lang_sanskrit_test/words.txt \
      exp/tri3/final.mdl exp/tri3/graph_sa/HCLG.fst \
      ark:decode/loglikes.ark "ark:|gzip -c > decode/out/lat.gz"

  lattice-best-path --acoustic-scale=$acwt \
      --word-symbol-table=data/lang_sanskrit_test/words.txt \
      "ark:gunzip -c decode/out/lat.gz|" ark,t:decode/out/tra.int

  utils/int2sym.pl -f 2- data/lang_sanskrit_test/words.txt decode/out/tra.int > decode/out/hyp.txt
fi

# 5) WER
if run 5; then
  echo "==================== WER (acwt=$acwt) ===================="
  compute-wer --text --mode=present ark:decode/ref_text ark:decode/out/hyp.txt
  echo "========================================================="
fi
