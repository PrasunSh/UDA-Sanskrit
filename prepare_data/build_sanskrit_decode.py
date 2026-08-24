#!/usr/bin/env python
"""
Prepare the Sanskrit-side inputs for WFST decoding (Step 2), all on the host:

  1. dict_sanskrit/   grapheme lexicon over the SAME phone set as the Hindi AM
                      (Sanskrit-only graphemes are dropped/mapped to Hindi phones)
  2. lm_corpus.txt    cleaned Sanskrit *train* transcripts -> bigram LM training text
  3. lm_vocab.txt     the LM/lexicon vocabulary (one word per line)
  4. decode/ref_text  cleaned Sanskrit *test* references in Kaldi 'utt words' format
                      (for compute-wer; test text is NOT used for the LM)

The phone set must match data/lang_hindi (the AM was trained on Hindi grapheme
pdf-ids), so every Sanskrit pronunciation uses only Hindi phones.
"""
import os, re, csv, unicodedata, collections

HI_DICT   = r"D:\uda_prep\kaldi\dict_hindi"
TRANS     = r"D:\transcripts_n2w\kb_data_clean_m4a\sanskrit"
OUT_DICT  = r"D:\uda_prep\kaldi\dict_sanskrit"
OUT_DEC   = r"D:\uda_prep\kaldi\decode"
TEST_MANIFEST = r"D:\uda_prep\manifests\sanskrit_test.tsv"   # our carved test split (utt + transcript)
WIKI_DIR  = r"D:\uda_prep\lm_wiki\extracted"                 # wikiextractor output (optional)
MIN_COUNT = 2   # prune words with corpus count < MIN_COUNT to <UNK> (keeps HCLG buildable)


def read_wiki(root):
    """Yield cleaned word-lists (sentences) from wikiextractor output."""
    if not os.path.isdir(root):
        return
    for dirpath, _, files in os.walk(root):
        for fn in files:
            with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                for line in f:
                    if line.startswith("<doc") or line.startswith("</doc") or not line.strip():
                        continue
                    for seg in re.split(r"[।॥\n]", line):     # split on danda / double danda
                        words = clean_words(seg)
                        if words:
                            yield words

DEVA_LO, DEVA_HI = 0x0900, 0x097F

# Sanskrit-only graphemes -> Hindi-set phone (or None = drop). Codepoints as ints.
GRAPHEME_MAP = {
    0x0951: None,     # udatta (Vedic stress) -> drop
    0x0952: None,     # anudatta            -> drop
    0x093D: None,     # avagraha (elision)  -> drop
    0x0946: 0x0947,   # short e sign  -> e sign
    0x094A: 0x094B,   # short o sign  -> o sign
    0x0944: 0x0943,   # vocalic RR sign -> vocalic R sign
    0x0934: 0x0933,   # LLLA -> LLA
    0x090C: 0x0932,   # vocalic L letter -> LA
    0x0961: 0x0932,   # vocalic LL letter -> LA
    0x0962: None,     # vocalic L sign -> drop
}


def load_hi_phones():
    return set(l.strip() for l in open(os.path.join(HI_DICT, "nonsilence_phones.txt"), encoding="utf-8") if l.strip())


def is_grapheme(ch):
    return DEVA_LO <= ord(ch) <= DEVA_HI and unicodedata.category(ch)[0] in ("L", "M")


def clean_words(text):
    text = unicodedata.normalize("NFC", text)
    out = []
    for w in text.split():
        w = "".join(ch for ch in w if is_grapheme(ch))
        if w:
            out.append(w)
    return out


def word_to_phones(word, hi_phones):
    """Grapheme -> Hindi phone, applying the map; skip anything still unknown."""
    phones = []
    for ch in word:
        cp = ord(ch)
        if cp in GRAPHEME_MAP:
            tgt = GRAPHEME_MAP[cp]
            if tgt is None:
                continue
            cp = tgt
        p = "U%04X" % cp
        if p in hi_phones:
            phones.append(p)
    return phones


def read_transcripts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rows.append((os.path.splitext(parts[0])[0], clean_words(parts[1])))
    return rows


def w(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    hi_phones = load_hi_phones()
    os.makedirs(OUT_DICT, exist_ok=True)
    os.makedirs(OUT_DEC, exist_ok=True)

    # --- our carved test split (exclude from LM to avoid contamination) ---
    test_refs = {}   # utt -> cleaned words
    with open(TEST_MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            test_refs[row["utt_id"]] = clean_words(row["transcript"])
    test_ids = set(test_refs)

    # --- LM corpus + vocab from Sanskrit TRAIN transcripts, EXCLUDING test utts ---
    train = read_transcripts(os.path.join(TRANS, "train", "transcription_n2w.txt"))
    corpus_lines, vocab = [], collections.Counter()
    excluded = 0
    for utt, words in train:
        if utt in test_ids:
            excluded += 1
            continue
        if not words:
            continue
        corpus_lines.append(" ".join(words))
        vocab.update(words)

    # --- add Sanskrit Wikipedia text to the LM corpus (if extracted) ---
    n_wiki = 0
    for words in read_wiki(WIKI_DIR):
        corpus_lines.append(" ".join(words))
        vocab.update(words)
        n_wiki += 1

    # --- decide the usable vocabulary: count >= MIN_COUNT AND a non-empty pron ---
    prons = {}
    dropped_nopron = 0
    for word, c in vocab.items():
        if c < MIN_COUNT:
            continue
        ph = word_to_phones(word, hi_phones)
        if not ph:
            dropped_nopron += 1
            continue
        prons[word] = ph
    usable = set(prons)

    # rewrite the corpus, mapping any non-usable word to <UNK> (so the LM models OOV)
    pruned = []
    for line in corpus_lines:
        pruned.append(" ".join(w_ if w_ in usable else "<UNK>" for w_ in line.split()))
    w(os.path.join(OUT_DEC, "lm_corpus.txt"), pruned)
    corpus_lines = pruned

    vocab_words = sorted(usable)
    w(os.path.join(OUT_DEC, "lm_vocab.txt"), vocab_words)

    # --- lexicon over the Hindi phone set ---
    lex = ["!SIL SIL", "<UNK> SPN"] + [f"{wd} {' '.join(prons[wd])}" for wd in vocab_words]
    import glob as _glob
    for f in _glob.glob(os.path.join(OUT_DICT, "lexiconp*.txt")):   # clear stale derived files
        os.remove(f)
    w(os.path.join(OUT_DICT, "lexicon.txt"), lex)
    dropped_words = dropped_nopron
    # reuse the Hindi phone inventory verbatim so ids line up with the AM
    for fn in ("nonsilence_phones.txt", "silence_phones.txt", "optional_silence.txt", "extra_questions.txt"):
        src = os.path.join(HI_DICT, fn)
        data = open(src, encoding="utf-8").read() if os.path.exists(src) else ""
        with open(os.path.join(OUT_DICT, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(data)

    # --- cleaned TEST reference for scoring (from our manifest) ---
    ref = sorted(f"{utt} {' '.join(words)}" for utt, words in test_refs.items() if words)
    w(os.path.join(OUT_DEC, "ref_text"), ref)

    # OOV rate of test words vs LM vocab (diagnostic; affects achievable WER)
    vset = set(vocab_words)
    tw = [wd for words in test_refs.values() for wd in words]
    oov = sum(1 for wd in tw if wd not in vset)
    print(f"LM corpus lines : {len(corpus_lines)}  (train {len(corpus_lines)-n_wiki} + wiki {n_wiki}; excluded {excluded} test utts)")
    print(f"vocabulary      : {len(vocab_words)} words  ({dropped_words} dropped: all-unknown graphemes)")
    print(f"lexicon entries : {len(lex)}")
    print(f"test references  : {len(ref)} utts")
    print(f"test-word OOV vs LM vocab: {oov}/{len(tw)} = {100*oov/max(len(tw),1):.1f}%")
    print(f"dict_sanskrit -> {OUT_DICT}")
    print(f"decode inputs -> {OUT_DEC}")


if __name__ == "__main__":
    main()
