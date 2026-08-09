#!/usr/bin/env python3
"""Economics of a pair-specialized acoustic disambiguator, measured before building it.

The user's idea: at decisions where the model hesitates between a known confusable pair
(des/les, est/et, a/à...), let a tiny waveform classifier pick. Such a classifier only fires
when {top1, top2} equals its pair, so the relevant statistics are, per pair, over ALL
decisions (correct and erroneous):

    fires      — how often {top1,top2} = pair (optionally under a margin gate)
    top1_right — sites where keeping the model's choice is correct
    top2_right — sites where the classifier would need to overturn it
    neither    — sites where both are wrong (classifier can't hurt or help WER there)

Break-even: a classifier with accuracy acc on the binary task changes WER by
    delta_errors = (1-acc) * top1_right - acc * top2_right
so it pays iff acc > top1_right / (top1_right + top2_right). That threshold, and the size of
top2_right (the max recoverable errors), decide whether the idea is worth an hour or a week.

Token-space alignment against the encoded raw reference — no normalization, no elision skip.

usage: pair_gate.py <ref.tsv> <topk_dump> <spm_model> [margin_gate]
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from margin_vs_error import align_ops

import sentencepiece as spm


def main():
    ref_tsv, dump_path, spm_path = sys.argv[1:4]
    gate = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9

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

    stats = collections.defaultdict(lambda: [0, 0, 0, 0])  # fires, top1_ok, top2_ok, neither
    n_utt = 0
    for f, toks in per_utt.items():
        if f not in refs or not toks:
            continue
        n_utt += 1
        hyp_ids = [t for t, _, _ in toks]
        ref_ids = sp.encode(refs[f])
        # token-level alignment: hyp position -> matching ref token (or None)
        ref_at = {}
        for op, ri, hj in align_ops(ref_ids, hyp_ids):
            if hj is not None:
                ref_at[hj] = ref_ids[ri] if ri is not None else None
        for k, (t1, t2, m) in enumerate(toks):
            if m >= gate:
                continue
            r = ref_at.get(k)
            key = tuple(sorted(sp.id_to_piece(x) for x in (t1, t2)))
            s = stats[key]
            s[0] += 1
            if r == t1:
                s[1] += 1
            elif r == t2:
                s[2] += 1
            else:
                s[3] += 1

    print(f"utterances: {n_utt}   margin gate: {gate}")
    print(f"{'pair':34s} {'fires':>5s} {'top1✓':>6s} {'top2✓':>6s} {'ni':>4s} "
          f"{'acc break-even':>15s}")
    rows = sorted(stats.items(), key=lambda kv: -kv[1][2])
    tot2 = 0
    for key, (fires, ok1, ok2, nei) in rows[:18]:
        if ok2 == 0:
            continue
        be = ok1 / (ok1 + ok2) if (ok1 + ok2) else float("nan")
        tot2 += ok2
        print(f"{str(key):34s} {fires:5d} {ok1:6d} {ok2:6d} {nei:4d} {100*be:14.0f} %")
    n_ref_tok = sum(len(sp.encode(refs[f])) for f in per_utt if f in refs)
    print(f"\nmax erreurs récupérables (Σ top2✓ des paires affichées) : {tot2} "
          f"tokens sur ~{n_ref_tok} ({100*tot2/n_ref_tok:.2f} % du flux token)")


if __name__ == "__main__":
    main()
