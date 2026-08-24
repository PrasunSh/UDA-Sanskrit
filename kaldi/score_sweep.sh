#!/usr/bin/env bash
# Rescore the existing lattices over a range of LM weights (LMWT) and word
# insertion penalties (WIP), compute WER for each, write to decode/wer_sweep.txt.
set -euo pipefail
export KALDI_ROOT=/opt/kaldi
cd "$(dirname "$0")"
. ./path.sh
. ./cmd.sh

lat="ark:gunzip -c decode/out/lat.gz|"
words=data/lang_sanskrit_test/words.txt
out=decode/wer_sweep.txt
: > $out

best=""
for lmwt in 14 15 16 17 18 19 20; do
  for wip in 1.0 1.5 2.0 2.5 3.0; do
    acwt=$(python3 -c "print(1.0/$lmwt)")
    hyp=decode/out/hyp_${lmwt}_${wip}.txt
    lattice-scale --inv-acoustic-scale=$lmwt "$lat" ark:- 2>/dev/null | \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- 2>/dev/null | \
      lattice-best-path --word-symbol-table=$words ark:- ark,t:- 2>/dev/null | \
      utils/int2sym.pl -f 2- $words > $hyp 2>/dev/null
    line=$(compute-wer --text --mode=present ark:decode/ref_text ark:$hyp 2>/dev/null | grep '%WER')
    echo "LMWT=$lmwt WIP=$wip | $line" | tee -a $out
    rm -f $hyp
  done
done
echo "done -> $out"
