#!/usr/bin/env python3
"""1-swap oracle: would taking the model's SECOND choice at a few low-confidence frames fix
the errors a beam search would have to fix?

Decides whether beam search is worth building. A beam explores alternatives inside one
decode (batch dim over hypotheses: the LM is weight-bound, so B=2 costs ~1.15x, not 2x),
which is the only phone-viable route to the 2.3 pt of headroom the 5-variant oracle exposed.
But a beam can only ever reach tokens that are in the top-k somewhere along the path. If
swapping in top-2 at the K lowest-margin frames does not reduce WER, the correct token is
simply not in the model's near-miss set and no amount of search will find it.

usage: swap_oracle.py <test.tsv> <hyp.tsv> <topk_dump> <spm_model> [K ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_wer import normalize, wer

import sentencepiece as spm


def detok(sp, ids):
    return sp.decode([i for i in ids if i > 3])


def main():
    test_tsv, hyp_path, dump_path, spm_path = sys.argv[1:5]
    Ks = [int(x) for x in sys.argv[5:]] or [1, 2, 3, 5, 10]
    sp = spm.SentencePieceProcessor(model_file=spm_path)

    refs = {}
    for line in open(test_tsv, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if len(c) >= 4:
            refs[c[1]] = c[3]

    # dump layout: the tokens of an utterance come FIRST, then its "# <file>" marker —
    # stt_eval writes the marker after printing the hypothesis. Treating the marker as a
    # header silently shifts every utterance's tokens onto its predecessor (it showed up as
    # a 112 % WER on a pipeline that scores 9 %).
    per_utt, cur = {}, []
    for line in open(dump_path):
        if line.startswith("# "):
            per_utt[line[2:].strip()] = cur
            cur = []
        else:
            a, b, m = line.split()
            cur.append((int(a), int(b), float(m)))

    base_e = base_n = 0
    swap_e = {k: 0 for k in Ks}
    scored = 0
    for f, toks in per_utt.items():
        if f not in refs or not toks:
            continue
        rw = normalize(refs[f])
        greedy = [t[0] for t in toks]
        S, I, D, n = wer(rw, normalize(detok(sp, greedy)))
        base_e += S + I + D
        base_n += n
        scored += 1
        # frames ordered by how close the decision was
        order = sorted(range(len(toks)), key=lambda i: toks[i][2])
        for K in Ks:
            best = S + I + D
            # independent single swaps among the K least-confident frames: the cheapest
            # possible proxy for what a beam explores jointly
            for i in order[:K]:
                alt = list(greedy)
                alt[i] = toks[i][1]
                s2, i2, d2, _ = wer(rw, normalize(detok(sp, alt)))
                best = min(best, s2 + i2 + d2)
            swap_e[K] += best

    print(f"utterances scored: {scored}")
    print(f"greedy WER       : {100*base_e/base_n:.2f} %")
    for K in Ks:
        print(f"  best 1-swap among {K:>2} least-confident frames: "
              f"{100*swap_e[K]/base_n:.2f} %")


if __name__ == "__main__":
    main()
