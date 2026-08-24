#!/usr/bin/env python
"""
Build a Witten-Bell smoothed **bigram** ARPA LM from a plain-text corpus
(one sentence per line, space-separated words). No SRILM needed.

  p(v|w)   = c(w,v) / (c(w) + T(w))                 for seen bigrams
  bow(w)   = T(w)   / (c(w) + T(w))                 backoff to unigram
  p1(v)    = c(v) / N                               unigram MLE (predicted tokens)
where c(w) = times w is used as a history, T(w) = # distinct successors of w.

Adds <s>/</s> sentence markers and a low-probability <UNK>. Output feeds Kaldi's
arpa2fst.  Usage:  python build_arpa.py decode/lm_corpus.txt decode/sanskrit.arpa
"""
import sys, math, collections

BOS, EOS, UNK = "<s>", "</s>", "<UNK>"
LOG0 = -99.0


def main(corpus_path, out_path):
    uni = collections.Counter()          # predicted-token counts (words + </s> + <UNK>)
    bi = collections.Counter()           # (w, v)
    hist = collections.Counter()         # times w used as history
    succ = collections.defaultdict(set)  # distinct successors of w

    n_sent = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            n_sent += 1
            toks = [BOS] + words + [EOS]
            for w in toks[1:]:
                uni[w] += 1
            for w, v in zip(toks[:-1], toks[1:]):
                bi[(w, v)] += 1
                hist[w] += 1
                succ[w].add(v)

    uni[UNK] += 1                          # reserve a little mass for OOV
    N = sum(uni.values())

    # unigram probs (log10). <s> is never predicted -> LOG0.
    vocab = sorted(uni)                    # predicted vocab: words, </s>, <UNK>
    log_p1 = {w: math.log10(uni[w] / N) for w in vocab}

    # unigram backoff weights: only histories have a bow; include <s>.
    hist_words = sorted(hist)              # words + <s>
    def bow(w):
        c, T = hist[w], len(succ[w])
        return T / (c + T) if (c + T) > 0 else 1.0

    # assemble the 1-gram section: every predicted word + <s>
    one_grams = {}                         # w -> (logp, logbow or None)
    for w in vocab:
        b = bow(w) if w in hist else 1.0
        one_grams[w] = (log_p1[w], math.log10(b) if b > 0 else LOG0)
    if BOS not in one_grams:
        b = bow(BOS)
        one_grams[BOS] = (LOG0, math.log10(b) if b > 0 else LOG0)

    # bigrams
    two_grams = []
    for (w, v), c in bi.items():
        denom = hist[w] + len(succ[w])
        two_grams.append((math.log10(c / denom), w, v))

    keys1 = sorted(one_grams)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\\data\\\n")
        f.write(f"ngram 1={len(keys1)}\n")
        f.write(f"ngram 2={len(two_grams)}\n\n")
        f.write("\\1-grams:\n")
        for w in keys1:
            lp, lb = one_grams[w]
            f.write(f"{lp:.6f}\t{w}\t{lb:.6f}\n")
        f.write("\n\\2-grams:\n")
        for lp, w, v in two_grams:
            f.write(f"{lp:.6f}\t{w} {v}\n")
        f.write("\n\\end\\\n")

    print(f"sentences={n_sent}  1-grams={len(keys1)}  2-grams={len(two_grams)}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
