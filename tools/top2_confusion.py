#!/usr/bin/env python3
"""Premise test for a waveform-level disambiguator at hesitant decisions.

The margin analysis showed the model HESITATES where it errs (median 4.45 vs 8.34). The
cheapest possible rescue would be: at low-margin decisions, let a tiny acoustic classifier
pick between the model's top-1 and top-2. That only works if, at error sites, the correct
token actually IS the top-2 — this script measures exactly that, from the night's
MOSHI_TOPK_DUMP, before any classifier is built.

For every substituted word made of a single token, we ask: does the top-2 token detokenize
to the reference word? Bonus tables: which (top1, top2) pairs dominate, and the margin
distribution of the fixable sites (a classifier can only be gated on margin, so the fixable
mass must live at low margin for the idea to ship).

usage: top2_confusion.py <ref.tsv> <topk_dump> <spm_model>
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_wer import normalize
from margin_vs_error import align_ops

import sentencepiece as spm


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

    n_sub = n_sub_1tok = n_top2_fix = 0
    fix_margins, nofix_margins = [], []
    pair_count = collections.Counter()
    for f, toks in per_utt.items():
        if f not in refs or not toks:
            continue
        # group tokens into words
        words, cur_w = [], []
        for k, (t, t2, m) in enumerate(toks):
            piece = sp.id_to_piece(t)
            if piece.startswith("▁") and cur_w:
                words.append(cur_w)
                cur_w = []
            cur_w.append((t, t2, m, k))
        if cur_w:
            words.append(cur_w)
        hyp_words = normalize(sp.decode([t for t, _, _ in toks]))
        ref_words = normalize(refs[f])
        if len(hyp_words) != len(words):
            # normalization can split tokens' words (elisions); skip utterances where the
            # token->word map is ambiguous rather than guess
            continue
        for op, ri, hj in align_ops(ref_words, hyp_words):
            if op != "sub":
                continue
            n_sub += 1
            w = words[hj]
            if len(w) != 1:
                continue
            n_sub_1tok += 1
            t1, t2, m, _ = w[0]
            alt = normalize(sp.decode([t2]))
            ok = len(alt) == 1 and alt[0] == ref_words[ri]
            if ok:
                n_top2_fix += 1
                fix_margins.append(m)
                pair_count[(hyp_words[hj], ref_words[ri])] += 1
            else:
                nofix_margins.append(m)

    print(f"substitutions alignées        : {n_sub}")
    print(f"  mot erroné = 1 token        : {n_sub_1tok}")
    print(f"  dont top-2 = le mot correct : {n_top2_fix} "
          f"({100*n_top2_fix/max(n_sub_1tok,1):.0f} % des 1-token, "
          f"{100*n_top2_fix/max(n_sub,1):.0f} % de toutes les substitutions)")

    def med(v):
        return sorted(v)[len(v)//2] if v else float("nan")
    print(f"\nmarge médiane des sites corrigeables par top-2 : {med(fix_margins):.2f}")
    print(f"marge médiane des sites NON corrigeables       : {med(nofix_margins):.2f}")
    if fix_margins:
        for thr in (1.0, 2.0, 4.0):
            cov = sum(1 for x in fix_margins if x < thr) / len(fix_margins)
            print(f"  part des corrigeables sous marge {thr}: {100*cov:.0f} %")

    print("\ntop paires (émis -> correct) corrigeables par top-2:")
    for (h, r), c in pair_count.most_common(12):
        print(f"  {c:3d}  {h!r} -> {r!r}")


if __name__ == "__main__":
    main()
