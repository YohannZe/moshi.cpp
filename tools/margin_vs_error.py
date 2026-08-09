#!/usr/bin/env python3
"""Is the model hesitant where it is wrong, or confidently wrong?

That single question separates the three candidate causes of the acoustically-distinct
substitutions (des->les, est->et, ...):

  hesitant when wrong  -> the acoustic evidence is weak: codec detail loss or frame-rate
                          smearing. Front-end / codec work is on the table.
  confident when wrong -> the language prior is overriding present evidence. Context and
                          decoding work is on the table, front-end work is not.

Method: reconstruct the emitted token stream from the MOSHI_TOPK_DUMP, detokenize
incrementally to map each token to the word it lands in, align words to the reference, and
compare the margin distribution of tokens inside correct words against tokens inside
substituted words.

usage: margin_vs_error.py <ref.tsv> <topk_dump> <spm_model>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_wer import normalize

import sentencepiece as spm


def align_ops(ref, hyp):
    """Levenshtein backtrace -> list of (op, ref_idx, hyp_idx)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                           dp[i-1][j-1] + (ref[i-1] != hyp[j-1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            out.append(("ok" if ref[i-1] == hyp[j-1] else "sub", i-1, j-1)); i, j = i-1, j-1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            out.append(("ins", None, j-1)); j -= 1
        else:
            out.append(("del", i-1, None)); i -= 1
    return out[::-1]


def main():
    ref_tsv, dump_path, spm_path = sys.argv[1:4]
    sp = spm.SentencePieceProcessor(model_file=spm_path)

    refs = {}
    for line in open(ref_tsv, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 4:
            refs[c[1]] = c[3]

    per_utt, cur = {}, []
    for line in open(dump_path):
        if line.startswith("# "):
            per_utt[line[2:].strip()] = cur
            cur = []
        else:
            a, b, m = line.split()
            cur.append((int(a), int(b), float(m)))

    ok_margins, sub_margins = [], []
    for f, toks in per_utt.items():
        if f not in refs or not toks:
            continue
        # word index for each token: a token starting a new word (SentencePiece marks it
        # with the U+2581 prefix) increments the counter
        widx, cw = [], -1
        for t, _, _ in toks:
            piece = sp.id_to_piece(t)
            if piece.startswith("▁"):
                cw += 1
            widx.append(max(cw, 0))
        hyp_words = normalize(sp.decode([t for t, _, _ in toks]))
        ref_words = normalize(refs[f])
        if not hyp_words:
            continue
        status = {}
        for op, _, hj in align_ops(ref_words, hyp_words):
            if hj is not None:
                status[hj] = op
        for (tok, _, margin), w in zip(toks, widx):
            st = status.get(w)
            if st == "ok":
                ok_margins.append(margin)
            elif st in ("sub", "ins"):
                sub_margins.append(margin)

    def stats(v):
        if not v:
            return "n/a"
        v = sorted(v)
        med = v[len(v)//2]
        p10 = v[len(v)//10]
        low = sum(1 for x in v if x < 2.0) / len(v)
        return (f"n={len(v):5d}  médiane={med:6.2f}  p10={p10:5.2f}  "
                f"part<2.0={100*low:5.1f} %")

    print("marges des tokens dans un mot CORRECT :", stats(ok_margins))
    print("marges des tokens dans un mot ERRONÉ  :", stats(sub_margins))
    if ok_margins and sub_margins:
        mo = sorted(ok_margins)[len(ok_margins)//2]
        ms = sorted(sub_margins)[len(sub_margins)//2]
        print(f"\nrapport des médianes (erroné / correct) = {ms/mo:.2f}")
        print("  << 1  -> le modèle HÉSITE là où il se trompe : évidence acoustique faible")
        print("  ~= 1  -> le modèle est CONFIANT à tort : le prior linguistique domine")


if __name__ == "__main__":
    main()
