#!/usr/bin/env python
"""
Build a grapheme-based Kaldi dict dir for Hindi, and clean the alignment text
consistently so every word in `text` is covered by the lexicon.

Grapheme model: after NFC normalization, each Devanagari *letter or combining
mark* (consonant, independent vowel, matra, virama, anusvara, visarga, nukta, ...)
is treated as one acoustic unit. Punctuation (danda etc.), digits and any
non-Devanagari characters are stripped. Each grapheme's phone symbol is its
Unicode code point as `U0915`, `U093E`, ... (ASCII-safe for Kaldi tooling).

This is the "get the pipeline running" unit set. It can later be swapped for a
phonemic G2P with schwa deletion (the paper's Hindi G2P) without touching the
rest of the recipe.

Outputs:
    <out_dict>/lexicon.txt            word  ->  U.... U....
    <out_dict>/nonsilence_phones.txt  one grapheme phone per line
    <out_dict>/silence_phones.txt     SIL, SPN
    <out_dict>/optional_silence.txt   SIL
    <out_dict>/extra_questions.txt    (empty; prepare_lang fills defaults)
Also rewrites <data_dir>/text (cleaned), backing up the original to text.raw.
"""
import argparse
import os
import unicodedata

DEVA_LO, DEVA_HI = 0x0900, 0x097F


def is_grapheme(ch):
    """Devanagari letter or combining mark (drops punctuation/digits/symbols)."""
    if not (DEVA_LO <= ord(ch) <= DEVA_HI):
        return False
    return unicodedata.category(ch)[0] in ("L", "M")


def clean_word(word):
    """NFC-normalize and keep only grapheme characters. May return ''."""
    word = unicodedata.normalize("NFC", word)
    return "".join(ch for ch in word if is_grapheme(ch))


def phone_of(ch):
    return "U%04X" % ord(ch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=r"D:\uda_prep\kaldi\data\hindi_train")
    ap.add_argument("--out-dict", default=r"D:\uda_prep\kaldi\dict_hindi")
    args = ap.parse_args()

    text_path = os.path.join(args.data_dir, "text")
    with open(text_path, encoding="utf-8") as f:
        raw_lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    vocab = {}                 # word -> [phones]
    graphemes = set()
    cleaned_lines = []
    dropped_utts = 0
    dropped_tokens = 0

    for ln in raw_lines:
        parts = ln.split(None, 1)
        if len(parts) < 2:
            dropped_utts += 1
            continue
        utt, text = parts[0], parts[1]
        words = []
        for w in text.split():
            cw = clean_word(w)
            if not cw:
                dropped_tokens += 1
                continue
            words.append(cw)
            if cw not in vocab:
                phones = [phone_of(ch) for ch in cw]
                vocab[cw] = phones
                graphemes.update(cw)
        if not words:
            dropped_utts += 1
            continue
        cleaned_lines.append(f"{utt} {' '.join(words)}")

    # --- write cleaned text (backup original once) ---
    raw_backup = text_path + ".raw"
    if not os.path.exists(raw_backup):
        os.replace(text_path, raw_backup)
    else:
        pass  # keep the first backup
    with open(text_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sorted(cleaned_lines)) + "\n")

    # --- write dict dir ---
    os.makedirs(args.out_dict, exist_ok=True)

    def w(fn, lines):
        with open(os.path.join(args.out_dict, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    lex = ["!SIL SIL", "<UNK> SPN"]
    lex += [f"{word} {' '.join(phones)}" for word, phones in sorted(vocab.items())]
    w("lexicon.txt", lex)
    w("nonsilence_phones.txt", sorted(phone_of(ch) for ch in graphemes))
    w("silence_phones.txt", ["SIL", "SPN"])
    w("optional_silence.txt", ["SIL"])
    w("extra_questions.txt", [])

    print(f"utts kept:        {len(cleaned_lines)}")
    print(f"utts dropped:     {dropped_utts}")
    print(f"empty tokens:     {dropped_tokens}")
    print(f"vocabulary size:  {len(vocab)}")
    print(f"grapheme phones:  {len(graphemes)}")
    print(f"dict dir:         {args.out_dict}")
    print(f"cleaned text:     {text_path}  (original -> text.raw)")


if __name__ == "__main__":
    main()
